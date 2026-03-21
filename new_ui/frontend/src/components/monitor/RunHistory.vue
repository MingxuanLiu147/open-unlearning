<template>
  <div>
    <div class="section-title">{{ $t('monitor.history') }}</div>
    <div v-if="runs.length === 0" class="empty">{{ $t('results.noData') }}</div>
    <div v-for="r in runs" :key="r.label" class="history-item ks-card" @click="$emit('select', r)">
      <div class="run-label">{{ r.label }}</div>
      <div class="run-meta">
        <span class="mode-badge" :class="r.mode">{{ r.mode }}</span>
        <span>{{ r.checkpoint }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { resultsApi } from '@/api'

defineEmits(['select'])
const runs = ref<any[]>([])

onMounted(async () => {
  runs.value = await resultsApi.list()
})
</script>

<style scoped>
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.history-item { margin-bottom: 6px; padding: 10px; }
.run-label { font-size: 12px; font-weight: 600; margin-bottom: 4px; word-break: break-all; }
.run-meta { display: flex; gap: 6px; align-items: center; font-size: 11px; color: var(--text-muted); }
.empty { color: var(--text-muted); font-size: 13px; padding: 20px; text-align: center; }
</style>
