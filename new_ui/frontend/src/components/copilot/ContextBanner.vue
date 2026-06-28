<template>
  <div class="context-banner">
    <span class="route-badge">{{ routeLabel }}</span>
    <span class="mode-badge" :class="experimentStore.mode" v-if="experimentStore.mode">
      {{ experimentStore.mode }}
    </span>
    <span class="ctx-tag" v-if="experimentStore.selectedModel">
      {{ experimentStore.selectedModel }}
    </span>
    <span class="ctx-tag" v-if="experimentStore.selectedTrainer">
      {{ experimentStore.selectedTrainer }}
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useExperimentStore } from '@/stores/experiment'
import { useI18n } from 'vue-i18n'

const route = useRoute()
const experimentStore = useExperimentStore()
const { t } = useI18n()

const routeLabel = computed(() => {
  const name = (route.name as string) || 'workshop'
  const key = `nav.${name}`
  return t(key)
})
</script>

<style scoped>
.context-banner {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border-color);
  flex-wrap: wrap;
  min-height: 36px;
}
.route-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--accent-primary-soft);
  color: var(--accent-primary);
}
.mode-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
}
.mode-badge.unlearn { background: var(--mode-unlearn-soft); color: var(--mode-unlearn); }
.mode-badge.inject  { background: var(--mode-inject-soft);  color: var(--mode-inject);  }
.mode-badge.edit    { background: var(--mode-edit-soft);    color: var(--mode-edit);    }
.mode-badge.eval    { background: var(--mode-eval-soft);    color: var(--mode-eval);    }
.ctx-tag {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 4px;
  background: var(--bg-card-hover);
  color: var(--text-secondary);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
