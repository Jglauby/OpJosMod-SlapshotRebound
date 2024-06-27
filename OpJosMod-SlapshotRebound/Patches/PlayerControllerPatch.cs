using AI;
using AI.Strategy;
using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using Michsky.UI.ModernUIPack;
using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.Playables;
using KeyCode = BepInEx.Unity.IL2CPP.UnityEngine.KeyCode;

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
        private static KeyCode pressedKey = KeyCode.P;
        private static KeyCode heldKey = KeyCode.C;

        [HarmonyPatch("Update")]
        [HarmonyPostfix]
        private static void UpdatePatch(PlayerController __instance)
        {
            if (Input.GetKeyInt(pressedKey) && alreadyPressed == false)
            {
                mls.LogInfo(pressedKey + " was pressed");
                OnButtonClick(__instance);
            }
            alreadyPressed = Input.GetKeyInt(pressedKey);

            if (Input.GetKeyInt(heldKey))
            {
                OnButtonHold(__instance);
            }
        }

        private static void OnButtonClick(PlayerController __instance)
        {
            //
        }

        private static void OnButtonHold(PlayerController __instance)
        {
            Puck puck = __instance.GetNearestPuck();
            if (puck != null)
            {
                Rigidbody rb = __instance.player.handsRigidbody;

                Vector3 playerPosition = rb.position;
                Vector3 puckPosition = puck.transform.position;
                Vector3 direction = (puckPosition - playerPosition).normalized;

                float speed = 15.0f;
                rb.AddForce(direction * speed, ForceMode.VelocityChange);
            }
            else
            {
                mls.LogError("no puck found");
            }
        }
    }
}
