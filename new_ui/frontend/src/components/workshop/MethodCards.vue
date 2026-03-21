<template>
  <div>
    <div class="section-title">{{ $t('workshop.method') }}</div>
    <div class="cards-grid">
      <div
        v-for="t in trainers"
        :key="t.name"
        class="ks-card"
        :class="{ selected: t.name === store.selectedTrainer }"
        @click="store.selectedTrainer = t.name"
      >
        <div class="card-header">
          <span class="card-name">{{ t.name.split('/').pop() }}</span>
          <span class="mode-badge" :class="store.mode">{{ store.mode }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const store = useExperimentStore()
const trainers = ref<any[]>([])

async function load() {
  trainers.value = await configApi.getTrainers(store.mode)
}

onMounted(load)
watch(() => store.mode, () => { store.selectedTrainer = ''; load() })
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
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 8px;
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.card-name {
  font-weight: 600;
  font-size: 13px;
}
</style>
