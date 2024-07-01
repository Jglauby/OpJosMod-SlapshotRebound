using BepInEx;
using BepInEx.Logging;
using BepInEx.Unity.IL2CPP;
using HarmonyLib;
using OpJosModSlapshotRebound.AIPlayer.Patches;

namespace OpJosModSlapshotRebound.AIPlayer
{
    [BepInPlugin(modGUID, modName, modVersion)]
    public class OpJosModSlapshotRebound : BasePlugin
    {
        private const string modGUID = "OpJosModSlapshotRebound.AIPlayer";
        private const string modName = "AIPlayer";
        private const string modVersion = "1.0.0";

        private readonly Harmony harmony = new Harmony(modGUID);

        private static OpJosModSlapshotRebound Instance;

        internal ManualLogSource mls;

        //not needed?
        public override void Load()
        {
            Loaded();
        }

        void Awake()
        {
            Loaded();
        }

        private void Loaded()
        {
            if (Instance == null)
            {
                Instance = this;
            }
            mls = BepInEx.Logging.Logger.CreateLogSource(modGUID);

            mls.LogInfo(modName + " has loaded");

            PlayerControllerPatch.SetLogSource(mls);
            harmony.PatchAll();
        }
    }
}
