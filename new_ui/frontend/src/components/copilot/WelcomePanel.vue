<template>
  <div class="welcome-panel">
    <div class="welcome-icon">
      <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M12 2a2 2 0 012 2c0 .74-.4 1.39-1 1.73v.27h1a7 7 0 017 7h-2a5 5 0 00-5-5h-4a5 5 0 00-5 5H3a7 7 0 017-7h1v-.27A2 2 0 0112 2z" fill="url(#wg)" stroke="none"/>
        <circle cx="10" cy="18" r="1.5" fill="url(#wg)" stroke="none"/>
        <circle cx="14" cy="18" r="1.5" fill="url(#wg)" stroke="none"/>
        <defs>
          <linearGradient id="wg" x1="0" y1="0" x2="24" y2="24">
            <stop offset="0%" stop-color="#667eea"/>
            <stop offset="100%" stop-color="#764ba2"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <h4 class="welcome-title">{{ $t('copilot.welcomeTitle') }}</h4>
    <p class="welcome-desc">{{ $t('copilot.welcomeDesc') }}</p>
    <div class="quick-actions">
      <SuggestionChip @click="$emit('ask', '帮我配置一个 unlearn 实验')">
        {{ $t('copilot.quickConfigExp') }}
      </SuggestionChip>
      <SuggestionChip @click="$emit('ask', '对比 GradAscent 和 NPO 的优劣')">
        {{ $t('copilot.quickMethodCompare') }}
      </SuggestionChip>
      <SuggestionChip @click="$emit('ask', '什么是知识遗忘？')">
        {{ $t('copilot.quickConcept') }}
      </SuggestionChip>
      <SuggestionChip @click="$emit('ask', '推荐合适的数据集')">
        {{ $t('copilot.quickDataset') }}
      </SuggestionChip>
    </div>
  </div>
</template>

<script setup lang="ts">
import SuggestionChip from './SuggestionChip.vue'

defineEmits<{
  (e: 'ask', question: string): void
}>()
</script>

<style scoped>
.welcome-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 40px 20px 24px;
  gap: 10px;
}
.welcome-icon {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 4px;
}
.welcome-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}
.welcome-desc {
  font-size: 13px;
  color: var(--text-muted);
  margin: 0;
  line-height: 1.5;
}
.quick-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px;
  margin-top: 8px;
}
</style>
