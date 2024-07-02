using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using System.Threading;
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
        private static List<AIInput> trainingData = new List<AIInput>();

        private static readonly string pluginDirectory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        private static readonly string modelPath = Path.Combine(pluginDirectory, "MLModel.zip");
        private static readonly string dataPath = Path.Combine(pluginDirectory, "trainingData.csv");

        public static float nextReward = 0f;

        private static Random random = new Random();
        private static float epsilon = 0.1f; // Start with a 10% chance of exploration
        private static float epsilonDecay = 0.995f; // Decay rate to reduce exploration over time
        private static float minEpsilon = 0.01f; // Minimum exploration probability


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
                    trainingData = new List<AIInput>
                    {
                        new AIInput { Features = new float[] { 1.0f, 0.0f }, Reward = 0.5f },
                        new AIInput { Features = new float[] { 0.0f, 1.0f }, Reward = 0.2f },
                    };
                    SaveTrainingData(dataPath, trainingData);
                }
            }
            catch (Exception ex)
            {
                mls.LogError("" + $"Error initializing ML: {ex.Message}");
            }
        }

        private static void RunAI()
        {
            try
            {
                float[] state = GetCurrentState();
                AIInput input = new AIInput { Features = state, Reward = nextReward };

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
                    mls.LogWarning("AI did not provide a valid action. Defaulting to move_towards_puck.");
                    action = "move_towards_puck"; // Default action if invalid
                }

                PerformAction(action);

                float reward = GetReward();
                trainingData.Add(new AIInput { Features = state, Reward = reward });

                if (trainingData.Count >= 100) // Train in batches of 100
                {
                    UpdateModel();
                    SaveTrainingData(dataPath, trainingData);
                    trainingData.Clear();
                }

                if (epsilon > minEpsilon)
                {
                    epsilon *= epsilonDecay;
                }

                mls.LogInfo("" + $"state: {string.Join(",", state)} action: {action} reward: {reward} epsilon: {epsilon}");
            }
            catch (Exception ex)
            {
                mls.LogError("" + $"Error running AI: {ex.Message}");
            }
        }


        private static string GetRandomAction()
        {
            var actions = new List<string> { "move_towards_puck", "shoot_left", "shoot_right", "spin_clockwise", "spin_counterclockwise" };
            return actions[random.Next(actions.Count)];
        }

        private static float[] GetCurrentState()
        {
            Vector3 puckLocation = GetPuckLocation();
            Vector3 playerLocation = GetPlayerLocation();
            Vector3 targetGoalLocation = GetTargetGoalLocation();

            float distanceFromPuck = GetDistanceFromPuck();
            float isCloseToPuck = distanceFromPuck < 1.0f ? 1.0f : 0.0f;
            float isAlignedWithGoal = Math.Abs((targetGoalLocation - playerLocation).x) < 1.0f ? 1.0f : 0.0f;

            return new float[] { isCloseToPuck, isAlignedWithGoal };
        }

        private static void PerformAction(string action)
        {
            switch (action)
            {
                case "move_towards_puck":
                    MoveTowardsDirection((GetPuckLocation() - GetPlayerLocation()).normalized);
                    break;
                case "shoot_left":
                    spinCounterClockwise();
                    break;
                case "shoot_right":
                    spinClockwise();
                    break;
                case "spin_clockwise":
                    spinClockwise();
                    break;
                case "spin_counterclockwise":
                    spinCounterClockwise();
                    break;
                default:
                    mls.LogError("" + $"Unknown action: {action}. Defaulting to move_towards_puck.");
                    MoveTowardsDirection((GetPuckLocation() - GetPlayerLocation()).normalized);
                    break;
            }
        }

        private static float GetReward()
        {
            var reward = nextReward;

            float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetTargetGoalLocation());
            float maxRewardDistance = 50.0f;
            float targetGoalReward = Mathf.Lerp(0.0f, 1.0f, 1.0f - Mathf.Clamp01(distanceToTargetGoal / maxRewardDistance));
            reward += targetGoalReward;

            if (targetGoalReward == 0)
            {
                float distanceToOwnGoal = Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation());
                float ownGoalPenalty = Mathf.Lerp(0.0f, 1.0f, Mathf.Clamp01(distanceToOwnGoal / maxRewardDistance));
                reward -= ownGoalPenalty;
            }

            //to encourage exploration?
            reward += UnityEngine.Random.Range(-0.05f, 0.05f);

            nextReward = 0;
            mls.LogInfo("" + $"gave reward: {reward}");
            return reward;
        }

        private static void UpdateModel()
        {
            try
            {
                var dataView = mlContext.Data.LoadFromEnumerable(trainingData);
                var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(AIInput.Reward))
                    .Append(mlContext.Transforms.Concatenate("Features", nameof(AIInput.Features)))
                    .Append(mlContext.Regression.Trainers.Sdca());

                trainedModel = pipeline.Fit(dataView);
                predictionEngine = mlContext.Model.CreatePredictionEngine<AIInput, AIOutput>(trainedModel);

                // Save the model
                mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
            }
            catch (Exception ex)
            {
                mls.LogError("" + $"Error updating model: {ex.Message}");
            }
        }

        public static void SetupAssemblyResolver()
        {
            AppDomain.CurrentDomain.AssemblyResolve += (sender, args) =>
            {
                mls.LogError("" + $"Failed to resolve assembly: {args.Name}");
                return null;
            };
        }

        private static void SaveTrainingData(string path, List<AIInput> data)
        {
            try
            {
                mls.LogInfo("" + $"Saving training data to {path}. Data count: {data.Count}");

                using (var writer = new StreamWriter(path, false)) // False to overwrite the file
                {
                    foreach (var item in data)
                    {
                        mls.LogInfo("" + $"Writing data: {item.Features[0]},{item.Features[1]},{item.Reward}");
                        writer.WriteLine($"{item.Features[0]},{item.Features[1]},{item.Reward}");
                    }
                }

                mls.LogInfo("Training data saved successfully.");
            }
            catch (IOException ex)
            {
                mls.LogError("" + $"Error saving training data: {ex.Message}");
            }
        }


        private static List<AIInput> LoadTrainingData(string path)
        {
            var data = new List<AIInput>();
            try
            {
                using (var reader = new StreamReader(path))
                {
                    while (!reader.EndOfStream)
                    {
                        var line = reader.ReadLine();
                        var values = line.Split(',');

                        if (values.Length != 3) // Ensure the line has the correct number of values
                        {
                            continue;
                        }

                        var features = new float[2];
                        features[0] = float.Parse(values[0]);
                        features[1] = float.Parse(values[1]);
                        var reward = float.Parse(values[2]);

                        data.Add(new AIInput { Features = features, Reward = reward });
                    }
                }
            }
            catch (Exception ex)
            {
                mls.LogError("" + $"Error loading training data: {ex.Message}");
            }
            return data;
        }

        #region get information
        private static Vector3 GetPuckLocation()
        {
            return localPlayer.GetNearestPuck().transform.position;
        }

        private static Vector3 GetPlayerLocation()
        {
            return localPlayer.playerRigidbody.transform.position;
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
            if (localPlayer.player.team == Team.Home)
            {
                return new Vector3(0, 0, -57);
            }

            return new Vector3(0, 0, 57);
        }

        private static Vector3 GetDefendingGoalLocation()
        {
            if (localPlayer.player.team == Team.Home)
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
                movement1(); // Move forward
            }
            else
            {
                movement2(); // Move backward
            }

            if (direction.x > 0)
            {
                movement4(); // Move right
            }
            else
            {
                movement3(); // Move left
            }
        }

        private static void movement1()
        {
            inputSimulator.Keyboard.KeyDown(forwardKey);
            inputSimulator.Keyboard.KeyUp(forwardKey);
        }

        private static void movement2()
        {
            inputSimulator.Keyboard.KeyDown(backwardKey);
            inputSimulator.Keyboard.KeyUp(backwardKey);
        }

        private static void movement3()
        {
            inputSimulator.Keyboard.KeyDown(leftKey);
            inputSimulator.Keyboard.KeyUp(leftKey);
        }

        private static void movement4()
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
                PlayerControllerPatch.nextReward = 250f;
                if (goalScored.ScorerID == PlayerControllerPatch.localPlayer.player.Id)
                {
                    PlayerControllerPatch.nextReward += 250f;
                }
            }
            else
            {
                PlayerControllerPatch.nextReward = -500f;
            }
        }
    }
    #endregion
}

public class AIInput
{
    [VectorType(2)]
    public float[] Features { get; set; }
    public float Reward { get; set; }
}

public class AIOutput
{
    public string Action { get; set; }
}
