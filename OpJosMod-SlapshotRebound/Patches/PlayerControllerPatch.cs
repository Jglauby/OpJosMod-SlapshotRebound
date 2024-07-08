using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Microsoft.ML;
using Microsoft.ML.Data;
using OpJosModSlapshotRebound.AIPlayer.Patches;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using UnityEngine;
using WindowsInput;
using WindowsInput.Native;
using KeyCode = BepInEx.Unity.IL2CPP.UnityEngine.KeyCode;
using Random = System.Random;

namespace OpJosModSlapshotRebound.AIPlayer.Patches
{
    public static class Constants
    {
        public const int ExpectedFeatures = 21; //needs to match the size of what we store
        public const bool isTraining = true;
        public const int DataSetSize = 90000;
        public const int MovementHeldTime = 2000; //how long holds down movement buttons in ms
    }

    public static class GlobalVars
    {
        public static string puckLastHitBy = string.Empty;
    }

    [HarmonyPatch(typeof(PlayerController))]
    internal class PlayerControllerPatch
    {
        public static ManualLogSource mls;
        public static void SetLogSource(ManualLogSource source)
        {
            mls = source;
            SetupAssemblyResolver();
        }

        private static bool aiEnabled = false;
        private static bool alreadyPressed = false;
        private static KeyCode toggleKey = KeyCode.R;

        public static PlayerController localPlayer = null;
        private static Game game = null;

        private static InputSimulator inputSimulator = new InputSimulator();
        private static VirtualKeyCode forwardKey = VirtualKeyCode.VK_W;
        private static VirtualKeyCode backwardKey = VirtualKeyCode.VK_S;
        private static VirtualKeyCode leftKey = VirtualKeyCode.VK_A;
        private static VirtualKeyCode rightKey = VirtualKeyCode.VK_D;
        private static VirtualKeyCode breakKey = VirtualKeyCode.SPACE;

        private static MLContext mlContext;
        private static PredictionEngine<AIInput, AIOutput> predictionEngine;
        private static ITransformer trainedModel;
        private static List<AISequence> trainingData = new List<AISequence>();
        private static List<AIInput> currentSequence = new List<AIInput>();

        private static readonly string pluginDirectory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        private static readonly string modelPath = Path.Combine(pluginDirectory, "MLModel.zip");
        private static readonly string dataPath = Path.Combine(pluginDirectory, "trainingData.csv");

        private static Vector3 previousPuckPosition = Vector3.zero;
        public static float nextReward = 0f;

        private static Random random = new Random();
        private static float epsilon = 0.7f; // Start with a 10% chance of exploration
        private static float epsilonDecay = 0.995f; // Decay rate to reduce exploration over time
        private static float minEpsilon = 0.05f; // Minimum exploration probability

        [HarmonyPatch("Update")]
        [HarmonyPostfix]
        private static void UpdatePatch(PlayerController __instance)
        {
            if (!__instance.player.local)
                return;

            if (localPlayer == null)
                localPlayer = __instance;

            if (game == null)
                game = AppManager.Instance.game;

            if (Input.GetKeyInt(toggleKey) && !alreadyPressed)
            {
                OnButtonClick();
            }
            alreadyPressed = Input.GetKeyInt(toggleKey);

            if (aiEnabled)
            {
                RunAI();
            }
        }

        private static void OnButtonClick()
        {
            aiEnabled = !aiEnabled;
            mls.LogInfo("AI " + (aiEnabled ? "enabled" : "disabled"));

            if (aiEnabled)
            {
                InitializeML();
            }
            else
            {
                UpdateModel();
                SaveTrainingData(dataPath, trainingData);
            }
        }

        private static void InitializeML()
        {
            try
            {
                mlContext = new MLContext();
                var blankFeatures = new float[Constants.ExpectedFeatures];
                blankFeatures[0] = 1.0f;

                if (File.Exists(modelPath))
                {
                    trainedModel = mlContext.Model.Load(modelPath, out _);
                    predictionEngine = mlContext.Model.CreatePredictionEngine<AIInput, AIOutput>(trainedModel);
                }
                else
                {
                    var initialData = new List<AIInput>
                    {
                        new AIInput { Features = blankFeatures, Reward = 0.5f },
                        new AIInput { Features = blankFeatures, Reward = 0.2f },
                    };
                    var dataView = mlContext.Data.LoadFromEnumerable(initialData);

                    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(AIInput.Reward))
                        .Append(mlContext.Transforms.Concatenate("Features", nameof(AIInput.Features)))
                        .Append(mlContext.Regression.Trainers.Sdca());

                    trainedModel = pipeline.Fit(dataView);
                    predictionEngine = mlContext.Model.CreatePredictionEngine<AIInput, AIOutput>(trainedModel);
                }

                if (File.Exists(dataPath))
                {
                    trainingData = LoadTrainingData(dataPath);
                }
                else
                {
                    trainingData = new List<AISequence>
                    {
                        new AISequence { Inputs = new List<AIInput>
                            {
                                new AIInput { Features = blankFeatures, Reward = 0.5f },
                                new AIInput { Features = blankFeatures, Reward = 0.2f },
                            }
                        }
                    };
                    UpdateModel();
                    SaveTrainingData(dataPath, trainingData);
                }
            }
            catch (Exception ex)
            {
                mls.LogError("Error initializing ML: " + ex.Message);
            }
        }

        private static void RunAI()
        {
            try
            {
                float[] state = GetCurrentState();
                AIInput input = new AIInput { Features = state, Reward = nextReward };

                currentSequence.Add(input);

                string action;

                if (random.NextDouble() < epsilon)
                {
                    // Exploration: take a random action
                    action = GetRandomAction();
                }
                else
                {
                    // Exploitation: use the model to predict the best action
                    AIOutput prediction = predictionEngine.Predict(input);
                    action = prediction?.Action ?? "do_nothing";
                }

                PerformAction(action);

                float reward = GetReward();
                previousPuckPosition = GetPuckLocation();
                input.Reward = reward;

                if (reward != 0)
                {
                    trainingData.Add(new AISequence { Inputs = new List<AIInput>(currentSequence) });
                    currentSequence.Clear();

                    if (trainingData.Count >= Constants.DataSetSize)
                    {
                        UpdateModel();
                        SaveTrainingData(dataPath, trainingData);

                        trainingData.RemoveRange(0, trainingData.Count / 2);
                    }

                    if (epsilon > minEpsilon)
                    {
                        epsilon *= epsilonDecay;
                    }

                    //mls.LogInfo("state: " + string.Join(",", state) + " action: " + action + " reward: " + reward + " epsilon: " + epsilon);
                }
            }
            catch (Exception ex)
            {
                mls.LogError("Error running AI: " + ex.Message);
            }
        }

        private static string GetRandomAction()
        {
            var actions = new List<string> {
                "do_nothing",
                "stop_move_press",
                "stop_move_release",
                "move_north_press",
                "move_north_release",
                "move_south_press",
                "move_south_release",
                "move_east_press",
                "move_east_release",
                "move_west_press",
                "move_west_release",
                "move_northeast_press",
                "move_northeast_release",
                "move_northwest_press",
                "move_northwest_release",
                "move_southeast_press",
                "move_southeast_release",
                "move_southwest_press",
                "move_southwest_release",
                "lift_stick", 
                "lower_stick", 
                "spin_clockwise", 
                "spin_counterclockwise",
                "spin_clockwisefast",
                "spin_counterclockwisefast"
            };
            return actions[random.Next(actions.Count)];
        }

        private static float[] GetCurrentState()
        {
            Vector3 puckLocation = GetPuckLocation();
            Vector3 playerLocation = GetPlayerLocation();
            Vector3 targetGoalLocation = GetTargetGoalLocation();
            Quaternion stickRotation = GetStickRotation();

            Vector3 stickRotationEuler = stickRotation.eulerAngles / 360.0f; // Normalize to [0, 1]
            List<Vector3> teammates = GetTeamMatesLocation();
            List<Vector3> opponents = GetOpponentsLocation();

            // Convert positions to a flat array (excluding Y values)
            List<float> state = new List<float>
            {
                puckLocation.x / 100.0f, puckLocation.z / 100.0f,
                playerLocation.x / 100.0f, playerLocation.z / 100.0f,
                targetGoalLocation.x / 100.0f, targetGoalLocation.z / 100.0f,
                stickRotationEuler.y
            };

            // Add teammate positions (excluding Y values)
            foreach (var teammate in teammates)
            {
                state.Add(teammate.x / 100.0f);
                state.Add(teammate.z / 100.0f);
            }

            // Add opponent positions (excluding Y values)
            foreach (var opponent in opponents)
            {
                state.Add(opponent.x / 100.0f);
                state.Add(opponent.z / 100.0f);
            }

            // Additional state information
            float distanceFromPuck = GetDistanceFromPuck() / 100.0f; // Assuming max distance can be 100 units
            float isCloseToPuck = distanceFromPuck < 0.01f ? 1.0f : 0.0f; // Normalized distance check
            float isAlignedWithGoal = Math.Abs((targetGoalLocation - playerLocation).x) < 1.0f ? 1.0f : 0.0f;

            state.Add(isCloseToPuck);
            state.Add(isAlignedWithGoal);

            return state.ToArray();
        }

        private static void PerformAction(string action)
        {
            //mls.LogInfo("performing action: " + action);
            switch (action)
            {
                case "do_nothing":
                    break;
                case "stop_move_press":
                    BreakPress();
                    break;
                case "stop_move_release":
                    BreakRelease();
                    break;
                case "move_north_press":
                    MoveForwardPress();
                    break;
                case "move_north_release":
                    MoveForwardRelease();
                    break;
                case "move_south_press":
                    MoveBackwardPress();
                    break;
                case "move_south_release":
                    MoveBackwardRelease();
                    break;
                case "move_east_press":
                    MoveRightPress();
                    break;
                case "move_east_release":
                    MoveRightRelease();
                    break;
                case "move_west_press":
                    MoveLeftPress();
                    break;
                case "move_west_release":
                    MoveLeftRelease();
                    break;
                case "move_northeast_press":
                    MoveRightPress();
                    MoveForwardPress();
                    break;
                case "move_northeast_release":
                    MoveRightRelease();
                    MoveForwardRelease();
                    break;
                case "move_northwest_press":
                    MoveForwardPress();
                    MoveLeftPress();
                    break;
                case "move_northwest_release":
                    MoveForwardRelease();
                    MoveLeftRelease();
                    break;
                case "move_southeast_press":
                    MoveBackwardPress();
                    MoveRightPress();
                    break;
                case "move_southeast_release":
                    MoveBackwardRelease();
                    MoveRightRelease();
                    break;
                case "move_southwest_press":
                    MoveBackwardPress();
                    MoveLeftPress();
                    break;
                case "move_southwest_release":
                    MoveBackwardRelease();
                    MoveLeftRelease();
                    break;
                case "lift_stick":
                    pickStickUp();
                    break;
                case "lower_stick":
                    putStickDown();
                    break;
                case "spin_clockwise":
                    spinClockwise();
                    break;
                case "spin_counterclockwise":
                    spinCounterClockwise();
                    break;
                case "spin_clockwisefast":
                    spinClockwiseFast();
                    break;
                case "spin_counterclockwisefast":
                    spinCounterClockwiseFast();
                    break;
                default:
                    mls.LogError("" + $"Unknown action: {action}. do nothing");
                    break;
            }
        }

        private static float GetReward()
        {
            float reward = 0f;

            //dont point reward at all when puck or player behind goal
            if (Math.Abs(GetPuckLocation().z) > 57 || Math.Abs(GetPlayerLocation().x) > 57)
                return 0f;

            //if hit puck away
            if (GlobalVars.puckLastHitBy == localPlayer.player.Username)
            {
                //now closer to target
                if (Vector3.Distance(GetTargetGoalLocation(), previousPuckPosition) > Vector3.Distance(GetTargetGoalLocation(), GetPuckLocation()))
                {
                    float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetTargetGoalLocation());
                    float targetGoalReward = 50 / distanceToTargetGoal;
                    reward += targetGoalReward;

                    reward += 5 / (GetPuckLocation().x + 1);
                }

                //now closer to own goal
                if (Vector3.Distance(GetDefendingGoalLocation(), previousPuckPosition) > Vector3.Distance(GetDefendingGoalLocation(), GetPuckLocation()))
                {
                    float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation());
                    float penalty = 50 / distanceToTargetGoal;
                    reward -= penalty * 2;

                    reward -= 5 / (GetPuckLocation().x + 1);
                }
            }

            if (TeamateHasPuck())
            {
                if (Vector3.Distance(GetPlayerLocation(), GetTargetGoalLocation()) < Vector3.Distance(GetPuckLocation(), GetTargetGoalLocation())) //player closer to goal than the puck is
                {
                    Vector3 directionToGoal = GetTargetGoalLocation() - GetPlayerLocation();
                    Vector3 directionToPuck = GetPuckLocation() - GetPlayerLocation();

                    float angle = Vector3.Angle(directionToGoal, directionToPuck);
                    if (angle > 30.0f) // Not in direct line (angle greater than 30 degrees)
                    {
                        //reward based on closeness to cetner of field
                        float proximityToCenter = 1 / (Mathf.Abs(GetPlayerLocation().x) + 1);
                        float baseReward = 10.0f * proximityToCenter;
                        reward += baseReward;

                        //give more reward if far from other players
                        List<Vector3> allPlayers = GetAllPlayersLocation();
                        float minPlayerDistance = Mathf.Infinity;

                        foreach (var player in allPlayers)
                        {
                            float distance = Vector3.Distance(GetPlayerLocation(), player);
                            if (distance < minPlayerDistance)
                                minPlayerDistance = distance;
                        }

                        if (minPlayerDistance > 3f) //if also not near other players add reward again
                        {
                            reward += baseReward;
                        }
                    }
                }
            }
            else if (OpponentHasPuck())
            {
                if (Vector3.Distance(GetPlayerLocation(), GetDefendingGoalLocation()) < Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation())) // player closer to goal than the puck is
                {
                    Vector3 directionToGoal = GetDefendingGoalLocation() - GetPlayerLocation();
                    Vector3 directionToPuck = GetPuckLocation() - GetPlayerLocation();

                    // Reward based on how "in the way" the player is
                    float angle = Vector3.Angle(directionToGoal, directionToPuck);
                    float angleReward = Mathf.Cos(angle * Mathf.Deg2Rad);

                    float baseReward = 20.0f * angleReward;

                    if (angle < 10)
                        reward += baseReward;
                }
            }

            //reward if close to puck
            if (Vector3.Distance(GetPuckLocation(), GetPlayerLocation()) < 10)
            {
                float distanceToPuck = Vector3.Distance(GetPuckLocation(), GetPlayerLocation());
                float targetReward = 5 / Math.Max(1, distanceToPuck);
            }

            reward += nextReward;

            // Encourage exploration with a small random factor
            if (reward > 0.09)
                reward += UnityEngine.Random.Range(-0.05f, 0.05f);

            PropagateRewards(reward);

            if (reward != 0)
            {
                if (reward > 0)
                    mls.LogWarning("Positive Feedback: " + reward);
                else
                    mls.LogInfo("Negative Feedback: " + reward);
            }

            nextReward = 0;
            return reward;
        }

        private static void PropagateRewards(float finalReward)
        {
            if (finalReward < 0.05)
                return;

            float decayFactor = 0.99f; // Decay factor for propagating rewards
            float reward = finalReward;

            int maxStepsBack = 1000;
            int steps = 0;

            for (int i = currentSequence.Count - 1; i >= 0; i--)
            {
                if (steps < maxStepsBack)
                {
                    currentSequence[i].Reward += reward;
                    reward *= decayFactor; 
                }

                steps++;
            }

            trainingData.Add(new AISequence { Inputs = new List<AIInput>(currentSequence) });
            currentSequence.Clear();
        }

        private static void UpdateModel()
        {
            try
            {
                var flattenedData = FlattenTrainingData(trainingData);

                // Check the dimensions of the feature vectors
                int expectedDimension = Constants.ExpectedFeatures; // The expected number of features
                foreach (var item in flattenedData)
                {
                    if (item.Features.Length != expectedDimension)
                    {
                        mls.LogError("" + $"Feature vector dimension mismatch. Expected: {expectedDimension}, Actual: {item.Features.Length}");
                        return;
                    }
                }

                var dataView = mlContext.Data.LoadFromEnumerable(flattenedData);
                var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(FlattenedAIInput.Reward))
                    .Append(mlContext.Transforms.Concatenate("Features", nameof(FlattenedAIInput.Features)))
                    .Append(mlContext.Regression.Trainers.Sdca());

                trainedModel = pipeline.Fit(dataView);
                predictionEngine = mlContext.Model.CreatePredictionEngine<AIInput, AIOutput>(trainedModel);

                // Save the model
                mls.LogWarning("Saving the model to " + modelPath);
                mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
                mls.LogWarning("Model saved successfully.");
            }
            catch (Exception ex)
            {
                mls.LogError("Error updating model: " + ex.Message);
            }
        }

        public static void SetupAssemblyResolver()
        {
            AppDomain.CurrentDomain.AssemblyResolve += (sender, args) =>
            {
                mls.LogError("Failed to resolve assembly: " + args.Name);
                return null;
            };
        }

        private static void SaveTrainingData(string path, List<AISequence> data)
        {
            try
            {
                mls.LogInfo("" + $"Saving training Data count: {data.Count}");
                using (var writer = new StreamWriter(path, false)) // False to overwrite the file
                {
                    foreach (var sequence in data)
                    {
                        foreach (var item in sequence.Inputs)
                        {
                            //mls.LogInfo("" + $"Writing data: {string.Join(",", item.Features)},{item.Reward}");
                            writer.WriteLine($"{string.Join(",", item.Features)},{item.Reward}");
                        }
                    }
                }

                //mls.LogInfo("Training data saved successfully.");
            }
            catch (IOException ex)
            {
                mls.LogError("Error saving training data: " + ex.Message);
            }
        }

        private static List<AISequence> LoadTrainingData(string path)
        {
            var data = new List<AISequence>();
            try
            {
                using (var reader = new StreamReader(path))
                {
                    var currentSequence = new List<AIInput>();
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(',');

                        if (values.Length == Constants.ExpectedFeatures) // Ensure the line has the correct number of values
                        {
                            continue;
                        }

                        var features = new float[Constants.ExpectedFeatures];
                        for (int i = 0; i < Constants.ExpectedFeatures; i++)
                        {
                            features[i] = float.Parse(values[i]);
                        }
                        var reward = float.Parse(values[Constants.ExpectedFeatures]);

                        currentSequence.Add(new AIInput { Features = features, Reward = reward });

                        if (reward != 0) // Sequence ends when a reward is given
                        {
                            data.Add(new AISequence { Inputs = new List<AIInput>(currentSequence) });
                            currentSequence.Clear();
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                mls.LogError("Error loading training data: " + ex.Message);
            }
            mls.LogInfo("loaded data with " +  data.Count + " records");
            return data;
        }

        private static List<FlattenedAIInput> FlattenTrainingData(List<AISequence> sequences)
        {
            var flattenedData = new List<FlattenedAIInput>();
            foreach (var sequence in sequences)
            {
                foreach (var input in sequence.Inputs)
                {
                    if (input.Features.Length != Constants.ExpectedFeatures)
                    {
                        mls.LogError("" + $"Feature vector dimension mismatch in FlattenTrainingData. Expected: {Constants.ExpectedFeatures}, Actual: {input.Features.Length}");
                        continue; // Skip this entry to prevent dimensionality issues
                    }
                    flattenedData.Add(new FlattenedAIInput { Features = input.Features, Reward = input.Reward });
                }
            }
            return flattenedData;
        }

        #region get information
        private static Vector3 GetPuckLocation()
        {
            var puck = localPlayer?.GetNearestPuck()?.transform?.position ?? Vector3.zero;
            if (puck == null)
            {
                mls.LogError("Puck location is null");
                return Vector3.zero;
            }
            return puck;
        }

        private static Vector3 GetPlayerLocation()
        {
            return localPlayer?.playerRigidbody?.transform?.position ?? Vector3.zero;
        }

        private static float GetDistanceFromPuck()
        {
            return Vector3.Distance(GetPlayerLocation(), GetPuckLocation());
        }

        public static Team GetPlayerTeam()
        {
            return localPlayer.player.team;
        }

        private static Vector3 GetTargetGoalLocation()
        {
            if (localPlayer?.player?.team == Team.Home)
            {
                return new Vector3(0, 0, -57);
            }

            return new Vector3(0, 0, 57);
        }

        private static Vector3 GetDefendingGoalLocation()
        {
            if (localPlayer?.player?.team == Team.Home)
            {
                return new Vector3(0, 0, 57);
            }

            return new Vector3(0, 0, -57);
        }

        private static List<Vector3> GetTeamMatesLocation()
        {
            List<Vector3> result = new List<Vector3>();
            foreach (Player player in game.Players.Values)
            {
                if (player.Team == GetPlayerTeam())
                {
                    result.Add(player.playerController.playerRigidbody.transform.position);
                }
            }

            return result;
        }

        private static List<Vector3> GetOpponentsLocation()
        {
            List<Vector3> result = new List<Vector3>();
            foreach (Player player in game.Players.Values)
            {
                if (player.Team != GetPlayerTeam())
                {
                    result.Add(player.playerController.playerRigidbody.transform.position);
                }
            }

            return result;
        }

        private static List<Vector3> GetAllPlayersLocation()
        {
            List<Vector3> result = new List<Vector3>();
            foreach (Player player in game.Players.Values)
            { 
                result.Add(player.playerController.playerRigidbody.transform.position);             
            }

            return result;
        }

        public static Quaternion GetStickRotation()
        {
            return localPlayer.handsRotatorRigidbody.rotation;
        }

        public static bool TeamateHasPuck()
        {
            //player too close
            if (Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) < 2)
                return false;

            //opponent too close
            foreach (Vector3 playerLoc in GetOpponentsLocation())
            {
                if (Vector3.Distance(playerLoc, GetPuckLocation()) < 2)
                    return false;
            }

            //teamamte close
            foreach (Vector3 playerLoc in GetTeamMatesLocation())
            {
                if (Vector3.Distance(playerLoc, GetPuckLocation()) < 2)
                    return true;
            }

            return false;
        }

        public static bool OpponentHasPuck()
        {
            //player too close
            if (Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) < 2)
                return false;

            //teamamte too close
            foreach (Vector3 playerLoc in GetTeamMatesLocation())
            {
                if (Vector3.Distance(playerLoc, GetPuckLocation()) < 2)
                    return false;
            }

            //opponent close
            foreach (Vector3 playerLoc in GetOpponentsLocation())
            {
                if (Vector3.Distance(playerLoc, GetPuckLocation()) < 2)
                    return true;
            }

            return false;
        }
        #endregion

        #region control player
        private static void MoveForwardPress()
        {
            inputSimulator.Keyboard.KeyDown(forwardKey);
        }

        private static void MoveForwardRelease()
        {
            inputSimulator.Keyboard.KeyUp(forwardKey);
        }

        private static void MoveBackwardPress()
        {
            inputSimulator.Keyboard.KeyDown(backwardKey);
        }

        private static void MoveBackwardRelease()
        {
            inputSimulator.Keyboard.KeyUp(backwardKey);
        }

        private static void MoveLeftPress()
        {
            inputSimulator.Keyboard.KeyDown(leftKey);
        }

        private static void MoveLeftRelease()
        {
            inputSimulator.Keyboard.KeyUp(leftKey);
        }

        private static void MoveRightPress()
        {
            inputSimulator.Keyboard.KeyDown(rightKey);
        }

        private static void MoveRightRelease()
        {
            inputSimulator.Keyboard.KeyUp(rightKey);
        }

        private static void BreakPress()
        {
            inputSimulator.Keyboard.KeyDown(breakKey);
        }

        private static void BreakRelease()
        {
            inputSimulator.Keyboard.KeyUp(breakKey);
        }

        private static void pickStickUp()
        {
            inputSimulator.Mouse.LeftButtonDown();
        }

        private static void putStickDown()
        {
            inputSimulator.Mouse.LeftButtonUp();
        }

        private static void spinClockwise()
        {
            int moveDistance = 100;
            inputSimulator.Mouse.MoveMouseBy(moveDistance, 0);
        }

        private static void spinCounterClockwise()
        {
            int moveDistance = -100;
            inputSimulator.Mouse.MoveMouseBy(moveDistance, 0);
        }

        private static void spinClockwiseFast()
        {
            Task.Run(() =>
            {
                int moveDistance = 500;
                for (int i = 0; i < 1000; i++)
                {
                    Task.Delay(1);
                    inputSimulator.Mouse.MoveMouseBy(moveDistance, 0);
                }
            });
        }

        private static void spinCounterClockwiseFast()
        {
            Task.Run(() =>
            {
                int moveDistance = -500;
                for (int i = 0; i < 1000; i++)
                {
                    Task.Delay(1);
                    inputSimulator.Mouse.MoveMouseBy(moveDistance, 0);
                }
            });
        }
        #endregion
    }

    #region extra feedback
    [HarmonyPatch(typeof(Game))]
    internal class GamePatch
    {
        [HarmonyPatch("OnGoalScored")]
        [HarmonyPostfix]
        private static void OnGoalScoredPatch(ref GoalScoredPacket goalScored)
        {
            if (goalScored.Team == PlayerControllerPatch.GetPlayerTeam())
            {
                PlayerControllerPatch.nextReward = 100f;
                if (goalScored.ScorerID == PlayerControllerPatch.localPlayer?.player?.Id)
                {
                    PlayerControllerPatch.nextReward += 2000f;
                }
            }
            else
            {
                PlayerControllerPatch.nextReward = -100f;
            }
        }

        [HarmonyPatch("Update")]
        [HarmonyPostfix]
        private static void UpdatePatch(Game __instance)
        {
            if (Constants.isTraining)
                __instance.MatchTimer = 500;
        }
    }

    [HarmonyPatch(typeof(Puck))]
    internal class PuckPatch
    {
        public static ManualLogSource mls;
        public static void SetLogSource(ManualLogSource source)
        {
            mls = source;
        }

        [HarmonyPatch("OnCollisionEnter")]
        [HarmonyPrefix]
        private static void OnCollisionEnterPatch(ref Collision collision)
        {
            //mls.LogInfo("collided with: " + collision.gameObject.name);
            if (collision.gameObject.name == "hands")
            {
                Player playerController = collision.gameObject.GetComponentInParent<Player>();
                if (playerController != null)
                {
                    //mls.LogMessage("Player hit puck: " + playerController.Username);
                    GlobalVars.puckLastHitBy = playerController.Username;
                }
                else
                {
                    mls.LogError("detected player hit puck is null");
                }
            }
            else if (collision.gameObject.name == "body")
            {
                Player playerController = collision.gameObject.GetComponentInParent<Player>();
                if (playerController != null)
                {
                    //mls.LogMessage("Player bumped puck: " + playerController.Username);
                    GlobalVars.puckLastHitBy = playerController.Username;
                }
                else
                {
                    mls.LogError("detected player hit puck is null");
                }
            }
        }
    }
    #endregion
}

public class AIInput
{
    [VectorType(Constants.ExpectedFeatures)]
    public float[] Features { get; set; }
    public float Reward { get; set; }
}

public class AIOutput
{
    public string Action { get; set; }
}

public class AISequence
{
    public List<AIInput> Inputs { get; set; }
}

public class FlattenedAIInput
{
    [VectorType(Constants.ExpectedFeatures)]
    public float[] Features { get; set; }
    public float Reward { get; set; }
}
