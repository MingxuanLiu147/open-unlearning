<template>
  <div>
    <div class="section-title">{{ $t('workshop.model') }}</div>
    <div class="cards-grid">
      <div
        v-for="m in models"
        :key="m.name"
        class="ks-card"
        :class="{ selected: m.name === store.selectedModel }"
        @click="store.selectedModel = m.name"
      >
        <div class="card-header">
          <span class="card-name">{{ m.name }}</span>
          <span class="modality-tag">{{ m.modality || 'text' }}</span>
        </div>
        <div class="card-meta">
          <span v-if="m.dtype">{{ m.dtype }}</span>
          <span v-if="m.attn">{{ m.attn }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const store = useExperimentStore()
const models = ref<any[]>([])

onMounted(async () => {
  models.value = await configApi.getModels()
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
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 4px;
}
.card-name {
  font-weight: 600;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.card-meta {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: var(--text-muted);
}
</style>
