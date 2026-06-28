<template>
  <div class="skill-bar" v-if="skills.length">
    <SuggestionChip
      v-for="s in skills"
      :key="s"
      :class="{ active: s === activeSkill }"
      @click="handleClick(s)"
    >{{ skillLabel(s) }}</SuggestionChip>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import SuggestionChip from './SuggestionChip.vue'
import type { SkillId } from '@/stores/agent'

const props = defineProps<{
  skills: SkillId[]
  activeSkill: SkillId | null
}>()

const emit = defineEmits<{
  (e: 'select', skillId: SkillId): void
  (e: 'clear'): void
}>()

const { t } = useI18n()

const SKILL_I18N_MAP: Record<string, string> = {
  config_wizard: 'copilot.skillConfigWizard',
  param_tuner: 'copilot.skillParamTuner',
  skill_recommender: 'copilot.skillSkillRecommender',
  edit_config_guide: 'copilot.skillEditConfigGuide',
  dataset_guide: 'copilot.skillDatasetGuide',
  forget_retain_advisor: 'copilot.skillForgetRetainAdvisor',
  result_analyzer: 'copilot.skillResultAnalyzer',
  next_step_advisor: 'copilot.skillNextStepAdvisor',
  concept_explainer: 'copilot.skillConceptExplainer',
  method_comparator: 'copilot.skillMethodComparator',
}

function skillLabel(id: SkillId): string {
  return t(SKILL_I18N_MAP[id] || id)
}

function handleClick(id: SkillId) {
  if (id === props.activeSkill) {
    emit('clear')
  } else {
    emit('select', id)
  }
}
</script>

<style scoped>
.skill-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 8px 12px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-copilot);
}
.skill-bar :deep(.suggestion-chip) {
  font-size: 11px;
  padding: 2px 8px;
}
.skill-bar :deep(.suggestion-chip.active) {
  background: var(--accent-primary);
  color: #fff;
  border-color: var(--accent-primary);
}
</style>
