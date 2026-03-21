<template>
  <div class="workshop">
    <!-- Mode selector -->
    <div class="mode-selector">
      <el-radio-group v-model="store.mode" size="default">
        <el-radio-button
          v-for="m in modes"
          :key="m.value"
          :value="m.value"
        >
          <span class="mode-badge" :class="m.value" style="background:transparent;">
            {{ $t(`workshop.modeOptions.${m.value}`) }}
          </span>
        </el-radio-button>
      </el-radio-group>
    </div>

    <!-- Three-column card area -->
    <el-row :gutter="16" class="config-area">
      <el-col :span="8">
        <ModelCards />
      </el-col>
      <el-col :span="8">
        <MethodCards />
      </el-col>
      <el-col :span="8">
        <DatasetCards />
        <div style="margin-top: 12px;">
          <EvalSelector />
        </div>
      </el-col>
    </el-row>

    <!-- Bottom: Params + Command + Run -->
    <div class="bottom-panel">
      <ParamsPanel />
      <CommandPreview />
    </div>
  </div>
</template>

<script setup lang="ts">
import { useExperimentStore } from '@/stores/experiment'
import ModelCards from '@/components/workshop/ModelCards.vue'
import MethodCards from '@/components/workshop/MethodCards.vue'
import DatasetCards from '@/components/workshop/DatasetCards.vue'
import EvalSelector from '@/components/workshop/EvalSelector.vue'
import ParamsPanel from '@/components/workshop/ParamsPanel.vue'
import CommandPreview from '@/components/workshop/CommandPreview.vue'

const store = useExperimentStore()

const modes = [
  { value: 'unlearn' },
  { value: 'inject' },
  { value: 'edit' },
  { value: 'eval' },
]
</script>

<style scoped>
.workshop {
  max-width: 1200px;
  margin: 0 auto;
}
.mode-selector {
  margin-bottom: 20px;
}
.config-area {
  margin-bottom: 16px;
}
.bottom-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  padding: 20px;
}
</style>
