using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using System;
using System.Collections.Generic;
using UnityEngine;
using WindowsInput;
using WindowsInput.Native;
using KeyCode = BepInEx.Unity.IL2CPP.UnityEngine.KeyCode;

namespace OpJosModSlapshotRebound.AIPlayer.Patches
{
    [HarmonyPatch(typeof(PlayerController))]
    internal class PlayerControllerPatch
    {
        public static ManualLogSource mls;
        public static void SetLogSource(ManualLogSource source)
        {
            mls = source;
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

        private static System.Random random = new System.Random();

        private static Dictionary<string, float> qTable = new Dictionary<string, float>();
        private static string previousState = "";
        private static string previousAction = "";
        private static float learningRate = 0.1f;
        private static float discountFactor = 0.9f;

        public static float nextReward = 0f;

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
        }

        private static void RunAI()
        {
            string state = GetCurrentState();
            string action = ChooseAction(state);

            PerformAction(action);

            float reward = GetReward();
            UpdateQTable(state, action, reward);

            previousState = state;
            previousAction = action;

            //mls.LogInfo("state: " + state + " action: " + action);
        }

        private static string GetCurrentState()
        {
            Vector3 puckLocation = GetPuckLocation();
            Vector3 playerLocation = GetPlayerLocation();
            Vector3 targetGoalLocation = GetTargetGoalLocation();

            float distanceFromPuck = GetDistanceFromPuck();
            bool isCloseToPuck = distanceFromPuck < 1.0f;
            bool isAlignedWithGoal = Math.Abs((targetGoalLocation - playerLocation).x) < 1.0f;

            return $"{isCloseToPuck}_{isAlignedWithGoal}";
        }

        private static string ChooseAction(string state)
        {
            if (!qTable.ContainsKey(state))
            {
                qTable[state] = 0.0f;
            }

            if (random.NextDouble() < 0.1) // epsilon-greedy exploration
            {
                string[] actions = { "move_towards_puck", "shoot_left", "shoot_right", "spin_clockwise", "spin_counterclockwise" };
                return actions[random.Next(actions.Length)];
            }

            return qTable[state] >= 0 ? "shoot_left" : "move_towards_puck";
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
            }
        }

        private static float GetReward()
        {
            var reward = 0.0f;
            if (nextReward != 0)
            {
                reward = nextReward;
            }

            //reward for hitting towards target goal
            float distanceToTargetGoal = Vector3.Distance(GetPuckLocation(), GetTargetGoalLocation());
            float maxRewardDistance = 50.0f; 
            float targetGoalReward = Mathf.Lerp(0.0f, 1.0f, 1.0f - Mathf.Clamp01(distanceToTargetGoal / maxRewardDistance)); 
            reward += targetGoalReward;

            //remove points when its headed towards defendign goal
            if (targetGoalReward == 0)
            {
                float distanceToOwnGoal = Vector3.Distance(GetPuckLocation(), GetDefendingGoalLocation());
                float ownGoalPenalty = Mathf.Lerp(0.0f, 1.0f, Mathf.Clamp01(distanceToOwnGoal / maxRewardDistance));
                reward -= ownGoalPenalty;
            }

            nextReward = 0;
            mls.LogInfo("gave reward: " + reward);
            return reward;
        }

        private static void UpdateQTable(string state, string action, float reward)
        {
            if (!qTable.ContainsKey(state))
            {
                qTable[state] = 0.0f;
            }

            float oldValue = qTable[state];
            float newValue = oldValue + learningRate * (reward + discountFactor * GetMaxQValue(state) - oldValue);
            qTable[state] = newValue;
        }

        private static float GetMaxQValue(string state)
        {
            // Placeholder for Q-value calculation
            // In a real scenario, this would return the maximum Q-value for the given state
            return 0.0f;
        }

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
                    PlayerControllerPatch.nextReward = 250f;
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
