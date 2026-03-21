<template>
  <el-tag
    class="suggestion-chip"
    effect="plain"
    round
    @click="$emit('click')"
    style="cursor: pointer;"
  >
    <slot />
  </el-tag>
</template>

<script setup lang="ts">
defineEmits(['click'])
</script>

<style scoped>
.suggestion-chip {
  margin: 2px 4px;
}
</style>
