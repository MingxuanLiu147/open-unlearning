<template>
  <div>
    <div class="section-title">{{ $t('workshop.eval') }}</div>
    <el-select
      v-model="store.selectedEval"
      :placeholder="$t('workshop.eval')"
      filterable
      clearable
      style="width: 100%;"
    >
      <el-option v-for="e in evals" :key="e" :label="e" :value="e" />
    </el-select>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const store = useExperimentStore()
const evals = ref<string[]>([])

async function load() {
  evals.value = await configApi.getEvals(store.mode)
}
onMounted(load)
watch(() => store.mode, load)
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
