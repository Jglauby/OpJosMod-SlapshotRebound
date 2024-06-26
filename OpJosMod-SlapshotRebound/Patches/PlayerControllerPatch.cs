using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Michsky.UI.ModernUIPack;
using System;
using System.Collections.Generic;
using System.Reflection;

namespace OpJosModSlapshotRebound.TestMod.Patches
{
    [HarmonyPatch(typeof(PlayerController))]
    internal class PlayerControllerPatch
    {
        private static ManualLogSource mls;
        public static void SetLogSource(ManualLogSource source)
        {
            mls = source;
        }

        private static bool alreadyPressed = false;
        private static KeyCode key = KeyCode.C;

        [HarmonyPatch("Update")]
        [HarmonyPostfix]
        private static void UpdatePatch(PlayerController __instance)
        {
            if (Input.GetKeyInt(key) && alreadyPressed == false)
            {
                mls.LogInfo(key + " was pressed");
                OnButtonClick(__instance);
            }

            alreadyPressed = Input.GetKeyInt(key);
        }

        private static void OnButtonClick(PlayerController __instnace) 
        {
            Puck puck = __instnace.GetNearestPuck();
        }
    }
}
