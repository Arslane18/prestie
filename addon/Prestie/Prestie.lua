-- Prestie: exports the character's state for the companion app.
--
-- WoW addons have no file I/O. The game serializes the global `PrestieDB` into
--   WTF/Account/<ACCOUNT>/SavedVariables/Prestie.lua
-- only on /reload, logout or exit. The snapshot is refreshed on every relevant
-- event, so whichever of those happens next writes an up-to-date state.
-- Schema read by the backend: src/prestie/character/state.py.

local ADDON_NAME = ...
local SCHEMA = 1

local frame = CreateFrame("Frame")

-- 11.x moved the spec functions to C_SpecializationInfo; keep a fallback.
local SpecInfo = C_SpecializationInfo or {}
local GetSpecIndex = SpecInfo.GetSpecialization or GetSpecialization
local GetSpecInfo = SpecInfo.GetSpecializationInfo or GetSpecializationInfo

local function readSpec()
  local index = GetSpecIndex and GetSpecIndex()
  if not index or not GetSpecInfo then
    return nil
  end
  local specID, name, _, _, role = GetSpecInfo(index)
  return { id = specID, name = name, role = role }
end

local function readHeroTalent()
  local ok, name = pcall(function()
    local subTreeID = C_ClassTalents.GetActiveHeroTalentSpec()
    local configID = C_ClassTalents.GetActiveConfigID()
    if not subTreeID or not configID then
      return nil
    end
    local info = C_Traits.GetSubTreeInfo(configID, subTreeID)
    return info and info.name
  end)
  return ok and name or nil
end

local function readObjectives(questID)
  local objectives = {}
  for _, objective in ipairs(C_QuestLog.GetQuestObjectives(questID) or {}) do
    table.insert(objectives, {
      text = objective.text,
      finished = objective.finished,
      fulfilled = objective.numFulfilled,
      required = objective.numRequired,
    })
  end
  return objectives
end

local function readQuests()
  local quests = {}
  for index = 1, C_QuestLog.GetNumQuestLogEntries() do
    local info = C_QuestLog.GetInfo(index)
    if info and not info.isHeader and not info.isHidden then
      table.insert(quests, {
        id = info.questID,
        title = info.title,
        level = info.level,
        complete = C_QuestLog.IsComplete(info.questID),
      })
    end
  end
  return quests
end

local function readActiveQuest()
  -- WoW has no "current quest"; the super-tracked one (the arrow) is the closest.
  local questID = C_SuperTrack.GetSuperTrackedQuestID()
  if not questID or questID == 0 then
    return nil
  end
  return {
    id = questID,
    title = C_QuestLog.GetTitleForQuestID(questID),
    objectives = readObjectives(questID),
  }
end

local function snapshot()
  local className, classFile = UnitClass("player")
  PrestieDB.snapshot = {
    capturedAt = time(),
    character = UnitName("player"),
    realm = GetRealmName(),
    level = UnitLevel("player"),
    class = { name = className, file = classFile },
    spec = readSpec(),
    heroTalent = readHeroTalent(),
    activeQuest = readActiveQuest(),
    quests = readQuests(),
  }
end

frame:SetScript("OnEvent", function(_, event, arg1)
  if event == "ADDON_LOADED" then
    if arg1 == ADDON_NAME then
      PrestieDB = PrestieDB or {}
      PrestieDB.schema = SCHEMA
      frame:UnregisterEvent("ADDON_LOADED")
    end
    return
  end
  if PrestieDB then
    snapshot()
  end
end)

frame:RegisterEvent("ADDON_LOADED")
for _, event in ipairs({
  "PLAYER_ENTERING_WORLD",
  "PLAYER_LEVEL_UP",
  "PLAYER_SPECIALIZATION_CHANGED",
  "TRAIT_CONFIG_UPDATED",
  "QUEST_ACCEPTED",
  "QUEST_REMOVED",
  "QUEST_TURNED_IN",
  "SUPER_TRACKING_CHANGED",
  "PLAYER_LOGOUT", -- last refresh before the game writes the file
}) do
  frame:RegisterEvent(event)
end

-- /prestie        show what the next write will contain
-- /prestie sync   /reload now, so the companion app sees the current state
SLASH_PRESTIE1 = "/prestie"
SlashCmdList.PRESTIE = function(message)
  if not PrestieDB then
    return
  end
  snapshot()
  if message == "sync" then
    ReloadUI()
    return
  end
  local snap = PrestieDB.snapshot
  print(string.format(
    "|cff00ccffPrestie|r: niv. %d %s %s (%s), quête suivie : %s, %d quêtes",
    snap.level,
    snap.spec and snap.spec.name or "sans spé",
    snap.class.name or "?",
    snap.heroTalent or "pas de talent héroïque",
    snap.activeQuest and snap.activeQuest.title or "aucune",
    #snap.quests
  ))
end
