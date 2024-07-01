using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
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

        private static bool alreadyPressed = false;
        private static KeyCode pressedKey = KeyCode.R;

        public static PlayerController localPlayer = null;
        private static Game game = null;

        private static InputSimulator inputSimulator = new InputSimulator();
        private static VirtualKeyCode forwardKey = VirtualKeyCode.VK_W;
        private static VirtualKeyCode backwardKey = VirtualKeyCode.VK_S;
        private static VirtualKeyCode leftKey = VirtualKeyCode.VK_A;
        private static VirtualKeyCode rightKey = VirtualKeyCode.VK_D;

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

            if (Input.GetKeyInt(pressedKey) && alreadyPressed == false)
            {
                OnButtonClick();
            }
            alreadyPressed = Input.GetKeyInt(pressedKey);
        }

        private static void OnButtonClick()
        {
            //toggle ai on and off here

            //this is a 3v3 hockey game and the goal is to score more points than your enemy team
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
            List<Vector3> result = null;
            foreach (Player player in game.Players.values)
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
            List<Vector3> result = null;
            foreach (Player player in game.Players.values)
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
            //-7 thorugh 7 x, for the width
            //-57 or 57 for z depending on which goal

            if (localPlayer.player.team == Team.Home)
            {
                return new Vector3(0, 0, -57);
            }

            else return new Vector3(0, 0, 57);
        }

        private static Vector3 GetDefendingGoalLocation()
        {
            if (localPlayer.player.team == Team.Home)
            {
                return new Vector3(0, 0, 57);
            }

            else return new Vector3(0, 0, -57);
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
    //useful for teeling ai what is good/bad
    [HarmonyPatch(typeof(Game))]
    internal class GamePatch
    {
        [HarmonyPatch("OnGoalScored")]
        [HarmonyPostfix]
        private static void OnGoalScoredPatch(ref GoalScoredPacket goalScored)
        {
            if (goalScored.Team == PlayerControllerPatch.GetPlayerTeam())
            {
                PlayerControllerPatch.mls.LogInfo("team scored");
                //goood!
                if (goalScored.ScorerID == PlayerControllerPatch.localPlayer.player.Id)
                {
                    //extra good
                    PlayerControllerPatch.mls.LogInfo("you scored!");
                }
            }
            else
            {
                //BAD
                PlayerControllerPatch.mls.LogInfo("enemy scored!");
            }
        }
    }
    #endregion
}
