<template>
  <div>
    <div class="section-title">{{ $t('workshop.model') }}</div>
    <el-select
      v-model="store.selectedModel"
      filterable
      clearable
      :placeholder="$t('workshop.selectModel')"
      style="width: 100%;"
    >
      <el-option-group :label="$t('workshop.modalityText')">
        <el-option
          v-for="m in textModels"
          :key="m.name"
          :label="formatLabel(m)"
          :value="m.name"
        />
      </el-option-group>
      <el-option-group :label="$t('workshop.modalityMultimodal')">
        <el-option
          v-for="m in mmModels"
          :key="m.name"
          :label="formatLabel(m)"
          :value="m.name"
        />
      </el-option-group>
    </el-select>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { configApi } from '@/api'
import { useExperimentStore } from '@/stores/experiment'

const store = useExperimentStore()
const models = ref<any[]>([])

const textModels = computed(() =>
  models.value.filter((m) => (m.modality || 'text') !== 'multimodal'),
)
const mmModels = computed(() =>
  models.value.filter((m) => (m.modality || 'text') === 'multimodal'),
)

function formatLabel(m: any) {
  const bits = [m.name]
  if (m.dtype) bits.push(String(m.dtype))
  return bits.join(' · ')
}

onMounted(async () => {
  models.value = await configApi.getModels()
  store.setModelCatalog(models.value)
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
</style>
