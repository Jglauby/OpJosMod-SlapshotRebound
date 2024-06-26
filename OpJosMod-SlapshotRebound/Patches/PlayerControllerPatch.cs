using AI;
using AI.Strategy;
using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Michsky.UI.ModernUIPack;
using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine.Playables;

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
        private static bool isBot = false;

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

        private static void OnButtonClick(PlayerController __instnace) //ai doenst turn on in real matches, just bot ones?
        {
            if (isBot)
            {
                isBot = false;
                __instnace.DisableAI();
            }
            else
            {
                isBot = true;
                __instnace.EnableAI();
                __instnace.aiController.difficulty = AIController.DifficultyEnum.Hard;
                __instnace.aiController.strategySelector = AIController.StrategyEnum.TeamplayStrategy;
            }
        }
    }
}
