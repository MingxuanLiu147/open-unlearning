<template>
  <div>
    <div class="section-title">{{ $t('workshop.method') }}</div>
    <el-select
      v-model="store.selectedTrainer"
      filterable
      clearable
      :placeholder="$t('workshop.selectMethod')"
      :disabled="store.mode === 'eval'"
      style="width: 100%;"
    >
      <el-option-group :label="$t('workshop.modalityText')">
        <el-option
          v-for="t in textTrainers"
          :key="t.name"
          :label="t.name.split('/').pop() || t.name"
          :value="t.name"
        />
      </el-option-group>
      <el-option-group :label="$t('workshop.modalityMultimodal')">
        <el-option
          v-for="t in mmTrainers"
          :key="t.name"
          :label="t.name.split('/').pop() || t.name"
          :value="t.name"
        />
      </el-option-group>
    </el-select>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, computed, onMounted } from 'vue'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const store = useExperimentStore()
const trainers = ref<any[]>([])

function isMM(t: any) {
  return (t.modality || 'text') === 'multimodal'
}

const textTrainers = computed(() =>
  trainers.value.filter((t) => !isMM(t)),
)
const mmTrainers = computed(() =>
  trainers.value.filter((t) => isMM(t)),
)

async function load() {
  trainers.value = await configApi.getTrainers(store.mode)
  pruneTrainer()
}

function pruneTrainer() {
  const wantMM = store.selectedModality === 'multimodal'
  const cur = trainers.value.find((t) => t.name === store.selectedTrainer)
  if (!cur) {
    store.selectedTrainer = ''
    return
  }
  const curMM = isMM(cur)
  if (curMM !== wantMM) store.selectedTrainer = ''
}

onMounted(load)
watch(() => store.mode, () => {
  store.selectedTrainer = ''
  load()
})
watch(() => store.selectedModality, () => pruneTrainer())
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
</style>
