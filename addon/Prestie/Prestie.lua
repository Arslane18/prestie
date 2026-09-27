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

-- Equipment ------------------------------------------------------------------
-- Raw game data only (slot ids, INVTYPE_* locations, GetItemStats keys): the
-- backend translates it (src/prestie/character/equipment.py).

local EQUIPMENT_SLOTS = { 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }
local FIRST_BAG, LAST_BAG = 0, 4 -- backpack + 4 bags (the reagent bag holds no gear)
local MAX_BAG_ITEMS = 40
local NOT_GEAR = {
  [""] = true,
  INVTYPE_NON_EQUIP_IGNORE = true,
  INVTYPE_BAG = true,
  INVTYPE_QUIVER = true,
  INVTYPE_BODY = true, -- shirt
  INVTYPE_TABARD = true,
}

local GetItemStats = C_Item.GetItemStats or GetItemStats
local GetItemInfoInstant = C_Item.GetItemInfoInstant or GetItemInfoInstant

-- Midnight hides some combat values as "secret values", which cannot be saved.
local function plain(value)
  if value == nil or (issecretvalue and issecretvalue(value)) then
    return nil
  end
  return type(value) == "number" and value or nil
end

local function call(fn, ...)
  if not fn then
    return nil
  end
  local ok, value = pcall(fn, ...)
  return ok and plain(value) or nil
end

local function readItem(link, location)
  -- |Hitem:<id>:<enchant>:<gem1>:<gem2>:<gem3>:<gem4>:...|h; the colour code
  -- before it (|cnIQ4:) also holds a colon, so isolate the item string first.
  local itemString = link:match("|Hitem:([^|]*)") or ""
  local id, enchant, gem1, gem2, gem3, gem4 = strsplit(":", itemString)
  local gems = 0
  for _, gem in ipairs({ gem1 or "", gem2 or "", gem3 or "", gem4 or "" }) do
    if tonumber(gem) then
      gems = gems + 1
    end
  end
  local stats = {}
  for key, value in pairs(GetItemStats(link) or {}) do
    stats[key] = plain(value)
  end
  return {
    id = tonumber(id),
    name = link:match("%[(.-)%]"),
    itemLevel = call(C_Item.GetCurrentItemLevel, location),
    enchant = tonumber(enchant),
    gems = gems,
    stats = stats,
  }
end

local function readEquipped()
  local items = {}
  for _, slot in ipairs(EQUIPMENT_SLOTS) do
    local link = GetInventoryItemLink("player", slot)
    if link then
      local item = readItem(link, ItemLocation:CreateFromEquipmentSlot(slot))
      item.slot = slot
      -- tells a two-handed weapon (empty off hand expected) from a one-handed one
      item.equipLoc = select(4, GetItemInfoInstant(link))
      table.insert(items, item)
    end
  end
  return items
end

local function readBags()
  local items = {}
  for bag = FIRST_BAG, LAST_BAG do
    for slot = 1, C_Container.GetContainerNumSlots(bag) do
      local link = C_Container.GetContainerItemLink(bag, slot)
      if link and #items < MAX_BAG_ITEMS then
        local _, _, subType, equipLoc = GetItemInfoInstant(link)
        if equipLoc and not NOT_GEAR[equipLoc] then
          local item = readItem(link, ItemLocation:CreateFromBagAndSlot(bag, slot))
          item.equipLoc = equipLoc
          item.subType = subType
          table.insert(items, item)
        end
      end
    end
  end
  return items
end

local function rating(combatRating, percent)
  return { rating = call(GetCombatRating, combatRating), percent = percent }
end

local function readStats()
  local overall, equipped = GetAverageItemLevel()
  local versatility = (call(GetCombatRatingBonus, CR_VERSATILITY_DAMAGE_DONE) or 0)
    + (call(GetVersatilityBonus, CR_VERSATILITY_DAMAGE_DONE) or 0)
  return {
    itemLevel = plain(overall),
    equippedItemLevel = plain(equipped),
    crit = rating(CR_CRIT_MELEE, call(GetCritChance)),
    haste = rating(CR_HASTE_MELEE, call(GetHaste)),
    mastery = rating(CR_MASTERY, call(GetMasteryEffect)),
    versatility = rating(CR_VERSATILITY_DAMAGE_DONE, versatility),
  }
end

local function readEquipment()
  return { equipped = readEquipped(), bags = readBags(), stats = readStats() }
end

local function snapshot()
  local className, classFile = UnitClass("player")
  -- Gear reading uses many game APIs: a failure there must not lose the rest,
  -- and is reported to the backend instead of silently dropped.
  local gearOk, equipment = pcall(readEquipment)
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
    equipment = gearOk and equipment or nil,
    equipmentError = not gearOk and tostring(equipment) or nil,
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
  "PLAYER_EQUIPMENT_CHANGED",
  "BAG_UPDATE_DELAYED",
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
    "|cff00ccffPrestie|r: niv. %d %s %s (%s), quête suivie : %s, %d quêtes, %s",
    snap.level,
    snap.spec and snap.spec.name or "sans spé",
    snap.class.name or "?",
    snap.heroTalent or "pas de talent héroïque",
    snap.activeQuest and snap.activeQuest.title or "aucune",
    #snap.quests,
    snap.equipment
        and string.format("ilvl %.1f, %d objets portés, %d dans les sacs",
          snap.equipment.stats.equippedItemLevel or 0,
          #snap.equipment.equipped, #snap.equipment.bags)
      or ("équipement illisible : " .. (snap.equipmentError or "?"))
  ))
end
