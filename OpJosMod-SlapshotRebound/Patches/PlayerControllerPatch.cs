using BepInEx.Logging;
using BepInEx.Unity.IL2CPP.UnityEngine;
using HarmonyLib;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using WindowsInput;
using WindowsInput.Native;
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

        private static InputSimulator inputSimulator = new InputSimulator();
        private static bool keysReleased = true;
        private static bool flipControls = false;
        private static VirtualKeyCode forwardKey = VirtualKeyCode.VK_W;
        private static VirtualKeyCode backwardKey = VirtualKeyCode.VK_S;
        private static VirtualKeyCode leftKey = VirtualKeyCode.VK_A;
        private static VirtualKeyCode rightKey = VirtualKeyCode.VK_D;
        private static List<Dictionary<VirtualKeyCode, float>> clickedKeysAt = new List<Dictionary<VirtualKeyCode, float>>();

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
            else
            {
                releaseKeys();
            }
        }

        private static void OnButtonClick(PlayerController __instance)
        {
            if (flipControls)
                flipControls = false;
            else 
                flipControls = true;
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

                if (Vector3.Distance(playerPosition, puckPosition) > 1.6)
                {
                    float angle = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg;
                    keysReleased = false;
                    //mls.LogInfo(angle);

                    if ((!flipControls && __instance.player.IsCameraFlipped()) || ((flipControls && !__instance.player.IsCameraFlipped())))
                    {
                        if (angle > -22.5 && angle <= 22.5)
                        {
                            rapidClickKey(forwardKey);
                        }
                        else if (angle > 22.5 && angle <= 67.5)
                        {
                            rapidClickKey(forwardKey);
                            rapidClickKey(rightKey);
                        }
                        else if (angle > 67.5 && angle <= 112.5)
                        {
                            rapidClickKey(rightKey);
                        }
                        else if (angle > 112.5 && angle <= 157.5)
                        {
                            rapidClickKey(rightKey);
                            rapidClickKey(backwardKey);
                        }
                        else if (angle > 157.5 || angle <= -157.5)
                        {
                            rapidClickKey(backwardKey);
                        }
                        else if (angle > -157.5 && angle <= -112.5)
                        {
                            rapidClickKey(backwardKey);
                            rapidClickKey(leftKey);
                        }
                        else if (angle > -112.5 && angle <= -67.5)
                        {
                            rapidClickKey(leftKey);
                        }
                        else if (angle > -67.5 && angle <= -22.5)
                        {
                            rapidClickKey(leftKey);
                            rapidClickKey(forwardKey);
                        }
                    }
                    else
                    {
                        if (angle > -22.5 && angle <= 22.5)
                        {
                            rapidClickKey(backwardKey);
                        }
                        else if (angle > 22.5 && angle <= 67.5)
                        {
                            rapidClickKey(leftKey);
                            rapidClickKey(backwardKey);
                        }
                        else if (angle > 67.5 && angle <= 112.5)
                        {
                            rapidClickKey(leftKey);
                        }
                        else if (angle > 112.5 && angle <= 157.5)
                        {
                            rapidClickKey(leftKey);
                            rapidClickKey(forwardKey);
                        }
                        else if (angle > 157.5 || angle <= -157.5)
                        {
                            rapidClickKey(forwardKey);
                        }
                        else if (angle > -157.5 && angle <= -112.5)
                        {
                            rapidClickKey(rightKey);
                            rapidClickKey(forwardKey);
                        }
                        else if (angle > -112.5 && angle <= -67.5)
                        {
                            rapidClickKey(rightKey);
                        }
                        else if (angle > -67.5 && angle <= -22.5)
                        {
                            rapidClickKey(rightKey);
                            rapidClickKey(backwardKey);
                        }
                    }
                }
                else
                {
                    mls.LogMessage("keys Released");
                    releaseKeys();
                }
            }
            else
            {
                mls.LogError("no puck found");
            }
        }

        private static void releaseKeys()
        {
            if (keysReleased)
                return;

            inputSimulator.Keyboard.KeyUp(forwardKey);
            inputSimulator.Keyboard.KeyUp(backwardKey);
            inputSimulator.Keyboard.KeyUp(leftKey);
            inputSimulator.Keyboard.KeyUp(rightKey);
            keysReleased = true;
        }

        private static void rapidClickKey(VirtualKeyCode key)
        {
            float timeBetween = 0f; // makes it go every time
            float currentTime = Time.time;
            Dictionary<VirtualKeyCode, float> currentDict = null;
            foreach (var dict in clickedKeysAt)
            {
                if (dict.ContainsKey(key))
                {
                    currentDict = dict;
                    break;
                }
            }

            if (currentDict == null)
            {
                currentDict = new Dictionary<VirtualKeyCode, float> {{ key, 0 }};
                clickedKeysAt.Add(currentDict);
            }

            //mls.LogInfo(currentTime + " | " + currentDict[key]);
            if (currentTime - currentDict[key] >= timeBetween)
            {
                //mls.LogMessage("key pressed: " + key);
                inputSimulator.Keyboard.KeyDown(key);
                currentDict[key] = currentTime;
            }

            inputSimulator.Keyboard.KeyUp(key);
        }
    }
}
