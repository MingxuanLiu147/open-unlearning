const BASE = ''

async function get<T = any>(url: string): Promise<T> {
  const res = await fetch(BASE + url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function post<T = any>(url: string, body?: any): Promise<T> {
  const res = await fetch(BASE + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function put<T = any>(url: string, body?: any): Promise<T> {
  const res = await fetch(BASE + url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

async function del<T = any>(url: string): Promise<T> {
  const res = await fetch(BASE + url, { method: 'DELETE' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const configApi = {
  getModels: () => get('/api/config/models'),
  getTrainers: (mode?: string) => get(`/api/config/trainers${mode ? `?mode=${mode}` : ''}`),
  getDatasets: (mode?: string) => get(`/api/config/datasets${mode ? `?mode=${mode}` : ''}`),
  getEvals: (mode?: string) => get(`/api/config/evals${mode ? `?mode=${mode}` : ''}`),
  getExperiments: (mode?: string) => get(`/api/config/experiments${mode ? `?mode=${mode}` : ''}`),
  getTrainerParams: (name: string) => get(`/api/config/trainer-params?name=${name}`),
  importConfig: (data: any) => post('/api/config/import', data),
  exportConfig: (data: any) => post('/api/config/export', data),
}

export const runnerApi = {
  start: (body: any) => post('/api/run/start', body),
  stop: () => post('/api/run/stop'),
  status: () => get('/api/run/status'),
  connectLog: (onData: (d: any) => void, onDone?: () => void) => {
    const es = new EventSource('/api/run/log')
    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.done) {
        es.close()
        onDone?.()
      } else {
        onData(data)
      }
    }
    es.onerror = () => { es.close(); onDone?.() }
    return es
  },
}

export const resultsApi = {
  list: () => get('/api/results/list'),
  get: (id: string) => get(`/api/results/${encodeURIComponent(id)}`),
  compare: (labels: string[]) => post('/api/results/compare', { labels }),
}

export const skillsApi = {
  list: () => get('/api/skills'),
  get: (id: string) => get(`/api/skills/${id}`),
  create: (data: any) => post('/api/skills', data),
  update: (id: string, data: any) => put(`/api/skills/${id}`, data),
  delete: (id: string) => del(`/api/skills/${id}`),
  run: (id: string) => post(`/api/skills/${id}/run`),
}

export const agentApi = {
  getConfig: () => get('/api/agent/config'),
  putConfig: (data: any) => put('/api/agent/config', data),
  test: (data: any) => post('/api/agent/test', data),
  streamChat: (
    messages: any[],
    context: any,
    onChunk: (text: string) => void,
    onDone: () => void,
  ) => {
    const ctrl = new AbortController()
    fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, context }),
      signal: ctrl.signal,
    }).then(async (res) => {
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() || ''
        for (const part of parts) {
          if (part.startsWith('data: ')) {
            const d = JSON.parse(part.slice(6))
            if (d.done) { onDone(); return }
            if (d.content) onChunk(d.content)
          }
        }
      }
      onDone()
    }).catch(() => onDone())
    return ctrl
  },
}

export const dataApi = {
  parse: (text: string) => post('/api/data/parse', { text }),
  validate: (records: any[]) => post('/api/data/validate', { records }),
}
