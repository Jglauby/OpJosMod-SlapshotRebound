using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;
using WindowsInput;
using WindowsInput.Native;
using KeyCode = BepInEx.Unity.IL2CPP.UnityEngine.KeyCode;
using Random = System.Random;

namespace OpJosModSlapshotRebound.AIPlayer.Patches
{
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
        private static float epsilon = 0.6f; // Start with a 10% chance of exploration
        private static float epsilonDecay = 0.9999f; // Decay rate to reduce exploration over time
        private static float minEpsilon = 0.1f; // Minimum exploration probability

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
        }

        private static void InitializeML()
        {
            try
            {
                mlContext = new MLContext();

                if (File.Exists(modelPath))
                {
                    trainedModel = mlContext.Model.Load(modelPath, out _);
                    predictionEngine = mlContext.Model.CreatePredictionEngine<AIInput, AIOutput>(trainedModel);
                }
                else
                {
                    var initialData = new List<AIInput>
                    {
                        new AIInput { Features = new float[] { 1.0f, 0.0f }, Reward = 0.5f },
                        new AIInput { Features = new float[] { 0.0f, 1.0f }, Reward = 0.2f },
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
                                new AIInput { Features = new float[] { 1.0f, 0.0f }, Reward = 0.5f },
                                new AIInput { Features = new float[] { 0.0f, 1.0f }, Reward = 0.2f },
                            }
                        }
                    };
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
                    action = prediction?.Action ?? "invalid_action";
                }

                if (action == "invalid_action" || string.IsNullOrEmpty(action))
                {
                    //mls.LogWarning("AI did not provide a valid action. Defaulting to move_towards_puck.");
                    action = "move_towards_puck"; // Default action if invalid
                }

                PerformAction(action);

                float reward = GetReward();
                previousPuckPosition = GetPuckLocation();
                input.Reward = reward;

                if (reward != 0)
                {
                    trainingData.Add(new AISequence { Inputs = new List<AIInput>(currentSequence) });
                    currentSequence.Clear();

                    if (trainingData.Count >= 1000) // Train in batches of 1000
                    {
                        mls.LogWarning("saving data");
                        UpdateModel();
                        SaveTrainingData(dataPath, trainingData);
                        trainingData.Clear();
                    }

                    if (epsilon > minEpsilon)
                    {
                        epsilon *= epsilonDecay;
                    }

                    mls.LogInfo("state: " + string.Join(",", state) + " action: " + action + " reward: " + reward + " epsilon: " + epsilon);
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
                "move_towards_puck",
                "move_north",
                "move_south",
                "move_east",
                "move_west",
                "move_northeast",
                "move_northwest",
                "move_southeast",
                "move_southwest",
                "lift_stick", 
                "lower_stick", 
                "spin_clockwise", 
                "spin_counterclockwise" 
            };
            return actions[random.Next(actions.Count)];
        }

        private static float[] GetCurrentState()
        {
            Vector3 puckLocation = GetPuckLocation();
            Vector3 playerLocation = GetPlayerLocation();
            Vector3 targetGoalLocation = GetTargetGoalLocation();

            float distanceFromPuck = GetDistanceFromPuck() / 100.0f; // Assuming max distance can be 100 units
            float isCloseToPuck = distanceFromPuck < 0.01f ? 1.0f : 0.0f; // Normalized distance check
            float isAlignedWithGoal = Math.Abs((targetGoalLocation - playerLocation).x) < 1.0f ? 1.0f : 0.0f;

            return new float[]
            {
                puckLocation.x / 100.0f, puckLocation.y / 100.0f, puckLocation.z / 100.0f,
                playerLocation.x / 100.0f, playerLocation.y / 100.0f, playerLocation.z / 100.0f,
                targetGoalLocation.x / 100.0f, targetGoalLocation.z / 100.0f, isCloseToPuck,
                isAlignedWithGoal
            };
        }

        private static void PerformAction(string action)
        {
            //mls.LogInfo("performing action: " + action);
            switch (action)
            {
                case "move_towards_puck":
                    MoveTowardsDirection((GetPuckLocation() - GetPlayerLocation()).normalized);
                    break;
                case "move_north":
                    MoveForward();
                    break;
                case "move_south":
                    MoveBackward();
                    break;
                case "move_east":
                    MoveRight();
                    break;
                case "move_west":
                    MoveLeft();
                    break;
                case "move_northeast":
                    MoveRight();
                    MoveForward();
                    break;
                case "move_northwest":
                    MoveForward();
                    MoveLeft();
                    break;
                case "move_southeast":
                    MoveBackward();
                    MoveRight();
                    break;
                case "move_southwest":
                    MoveBackward();
                    MoveLeft();
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
                default:
                    //mls.LogError("" + $"Unknown action: {action}. Defaulting to move_towards_puck.");
                    MoveTowardsDirection((GetPuckLocation() - GetPlayerLocation()).normalized);
                    break;
            }
        }

        private static float GetReward()
        {
            float reward = nextReward;

            //punish for puck moving away from target goal
            float maxRewardDistance = 100.0f; // Adjusted to 100 units for normalization
            float distanceToOwnGoal = Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation());
            float ownGoalPenalty = Mathf.Lerp(0.0f, 1.0f, Mathf.Clamp01(distanceToOwnGoal / maxRewardDistance));
            reward -= ownGoalPenalty;

            //reward puck going towards target goal
            if (Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) < 3)
            {
                float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetTargetGoalLocation());
                float targetGoalReward = Mathf.Lerp(0.0f, 1.0f, 1.0f - Mathf.Clamp01(distanceToTargetGoal / maxRewardDistance));
                reward += targetGoalReward;
            }

            //punish for moving towards own goal
            if (Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) < 3)
            {
                float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation());
                float penalty = Mathf.Lerp(0.0f, 1.0f, 1.0f - Mathf.Clamp01(distanceToTargetGoal / maxRewardDistance));
                reward -= penalty*2;
            }

            //reward for hitting puck towards target goal
            if (Vector3.Distance(GetPlayerLocation(), previousPuckPosition) < 2 &&
                Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) > 2 &&
                (Vector3.Distance(GetTargetGoalLocation(), previousPuckPosition) > Vector3.Distance(GetTargetGoalLocation(), GetPuckLocation()))) 
            {
                reward += 10f;
            }

            //punish for hitting towards own goal
            if (Vector3.Distance(GetPlayerLocation(), previousPuckPosition) < 2 &&
                Vector3.Distance(GetPlayerLocation(), GetPuckLocation()) > 2 &&
                (Vector3.Distance(GetTargetGoalLocation(), previousPuckPosition) > Vector3.Distance(GetDefendingGoalLocation(), GetPuckLocation())))
            {
                reward -= 20f;
            }

            // Encourage exploration with a small random factor
            reward += UnityEngine.Random.Range(-0.05f, 0.05f);

            //look into retroactive reward giving
            //like the trainingData, like maybe go back 100 records and add some sort of reward scalign down with how far back it is

            nextReward = 0;
            return reward;
        }

        private static void UpdateModel()
        {
            try
            {
                var flattenedData = FlattenTrainingData(trainingData);

                // Check the dimensions of the feature vectors
                int expectedDimension = 10; // The expected number of features
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
                //mls.LogInfo("" + $"Saving training data to {path}. Data count: {data.Count}");
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

                        if (values.Length != 11) // Ensure the line has the correct number of values
                        {
                            continue;
                        }

                        var features = new float[10];
                        for (int i = 0; i < 10; i++)
                        {
                            features[i] = float.Parse(values[i]);
                        }
                        var reward = float.Parse(values[10]);

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
            return data;
        }

        private static List<FlattenedAIInput> FlattenTrainingData(List<AISequence> sequences)
        {
            var flattenedData = new List<FlattenedAIInput>();
            foreach (var sequence in sequences)
            {
                foreach (var input in sequence.Inputs)
                {
                    if (input.Features.Length != 10)
                    {
                        mls.LogError("" + $"Feature vector dimension mismatch in FlattenTrainingData. Expected: 10, Actual: {input.Features.Length}");
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
        #endregion

        #region control player
        private static void MoveTowardsDirection(Vector3 direction)
        {
            if (direction.z > 0)
            {
                MoveForward(); // Move forward
            }
            else
            {
                MoveBackward(); // Move backward
            }

            if (direction.x > 0)
            {
                MoveRight(); // Move right
            }
            else
            {
                MoveLeft(); // Move left
            }
        }

        private static void MoveForward()
        {
            inputSimulator.Keyboard.KeyDown(forwardKey);
            inputSimulator.Keyboard.KeyUp(forwardKey);
        }

        private static void MoveBackward()
        {
            inputSimulator.Keyboard.KeyDown(backwardKey);
            inputSimulator.Keyboard.KeyUp(backwardKey);
        }

        private static void MoveLeft()
        {
            inputSimulator.Keyboard.KeyDown(leftKey);
            inputSimulator.Keyboard.KeyUp(leftKey);
        }

        private static void MoveRight()
        {
            inputSimulator.Keyboard.KeyDown(rightKey);
            inputSimulator.Keyboard.KeyUp(rightKey);
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
                PlayerControllerPatch.nextReward = 2500f;
                if (goalScored.ScorerID == PlayerControllerPatch.localPlayer?.player?.Id)
                {
                    PlayerControllerPatch.nextReward += 2500f;
                }
            }
            else
            {
                PlayerControllerPatch.nextReward = -5000f;
            }
        }
    }
    #endregion
}

public class AIInput
{
    [VectorType(10)]
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
    [VectorType(10)]
    public float[] Features { get; set; }
    public float Reward { get; set; }
}
