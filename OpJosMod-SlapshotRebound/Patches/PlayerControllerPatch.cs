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
        private static KeyCode pressedKey = KeyCode.R;
        private static KeyCode heldKey = KeyCode.C;
        private static bool controlPuck = false;

        [HarmonyPatch("Update")]
        [HarmonyPostfix]
        private static void UpdatePatch(PlayerController __instance)
        {
            if (!__instance.player.local)
                return;

            if (Input.GetKeyInt(pressedKey) && alreadyPressed == false)
            {
                OnButtonClick(__instance);
            }
            alreadyPressed = Input.GetKeyInt(pressedKey);

            if (Input.GetKeyInt(heldKey))
            {
                OnButtonHold(__instance);
            }

            HandlePuckControls(__instance);
        }

        private static void OnButtonClick(PlayerController __instance)
        {
            Puck puck = __instance.GetNearestPuck();
            if (controlPuck)
            {
                mls.LogInfo("Stop controlling puck");
                controlPuck = false;
            }
            else
            {

                mls.LogInfo("Start controlling puck");
                controlPuck = true;
            }
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

        private static void HandlePuckControls(PlayerController __instance)
        {
            if (!controlPuck)
                return;

            //stop player movment

            Puck puck = __instance.GetNearestPuck();
            Rigidbody rb = puck.PuckRigidbody;
            Vector3 puckPosition = rb.transform.position;

            Vector3 direction = new Vector3(0, 0, 0);

            if (__instance.player.IsCameraFlipped())
            {
                if (Input.GetKeyInt(KeyCode.W))
                {
                    direction = Vector3.forward;
                }
                if (Input.GetKeyInt(KeyCode.A))
                {
                    direction = Vector3.left;
                }
                if (Input.GetKeyInt(KeyCode.S))
                {
                    direction = Vector3.back;
                }
                if (Input.GetKeyInt(KeyCode.D))
                {
                    direction = Vector3.right;
                }
            }
            else
            {
                if (Input.GetKeyInt(KeyCode.W))
                {
                    direction = Vector3.back;
                }
                if (Input.GetKeyInt(KeyCode.A))
                {
                    direction = Vector3.right;
                }
                if (Input.GetKeyInt(KeyCode.S))
                {
                    direction = Vector3.forward;
                }
                if (Input.GetKeyInt(KeyCode.D))
                {
                    direction = Vector3.left;
                }
            }

            float speed = 0.5f;
            rb.AddForce(direction * speed, ForceMode.VelocityChange);
        }
    }
}
