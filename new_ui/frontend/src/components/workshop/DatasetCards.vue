<template>
  <div>
    <div class="section-title">{{ $t('workshop.dataset') }}</div>
    <div v-for="(items, group) in datasets" :key="group" class="ds-group">
      <div class="group-label">{{ groupLabel(group as string) }}</div>
      <el-select
        :model-value="store.selectedDatasets[group as string] || ''"
        @update:model-value="(v: string) => store.selectedDatasets[group as string] = v"
        :placeholder="`Select ${group}`"
        filterable
        clearable
        style="width: 100%;"
      >
        <el-option v-for="ds in items" :key="ds" :label="ds" :value="ds" />
      </el-select>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const { t } = useI18n()
const store = useExperimentStore()
const datasets = ref<Record<string, string[]>>({})

async function load() {
  datasets.value = await configApi.getDatasets(store.mode)
}

function groupLabel(g: string): string {
  const map: Record<string, string> = {
    forget: t('workshop.forget'),
    retain: t('workshop.retain'),
    edit: t('workshop.editDs'),
    train: t('workshop.trainDs'),
    all: t('workshop.dataset'),
  }
  return map[g] || g
}

onMounted(load)
watch(() => store.mode, () => { store.selectedDatasets = {}; load() })
watch(() => store.datasetListVersion, () => { load() })
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
.ds-group {
  margin-bottom: 10px;
}
.group-label {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 4px;
}
</style>
