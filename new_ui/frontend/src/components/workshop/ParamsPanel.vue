<template>
  <div class="params-panel">
    <div class="section-title">{{ $t('workshop.params') }}</div>
    <el-form label-position="top" size="small">
      <el-row :gutter="12">
        <el-col :span="8">
          <el-form-item :label="$t('workshop.taskName')">
            <el-input v-model="store.taskName" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.seed')">
            <el-input-number v-model="store.seed" :min="0" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.gpu')">
            <el-input v-model="store.gpu" placeholder="0" />
          </el-form-item>
        </el-col>
      </el-row>
      <el-row :gutter="12">
        <el-col :span="8">
          <el-form-item :label="$t('workshop.learningRate')">
            <el-input v-model="store.params.learning_rate" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.epochs')">
            <el-input-number v-model="store.params.num_epochs" :min="1" :max="100" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.batchSize')">
            <el-input-number v-model="store.params.batch_size" :min="1" style="width:100%" />
          </el-form-item>
        </el-col>
      </el-row>
      <el-row :gutter="12">
        <el-col :span="8">
          <el-form-item :label="$t('workshop.gradAccum')">
            <el-input-number v-model="store.params.gradient_accumulation_steps" :min="1" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.maxLength')">
            <el-input-number v-model="store.params.max_length" :min="64" :step="64" style="width:100%" />
          </el-form-item>
        </el-col>
        <el-col :span="8">
          <el-form-item :label="$t('workshop.warmup')">
            <el-input-number v-model="store.params.warmup_ratio" :min="0" :max="1" :step="0.05" :precision="2" style="width:100%" />
          </el-form-item>
        </el-col>
      </el-row>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { useExperimentStore } from '@/stores/experiment'
const store = useExperimentStore()
</script>

<style scoped>
.params-panel {
  margin-top: 8px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
</style>
