/**
 * 媒体下载：成片优先走 nanobot workplace 代理，避免跨域与 <a download> 失效。
 */

export interface DownloadMediaOptions {
  /** 媒体资源 URL（同域直链下载时使用） */
  url: string
  /** 自定义下载文件名 */
  fileName?: string
  /** workplace session key；与 token 同时提供时走 /download/final 代理 */
  sessionKey?: string
  /** WebUI Bearer token */
  token?: string
  onError?: (error: Error) => void
}

let isDownloading = false

const isCrossOrigin = (url: string): boolean => {
  try {
    return window.location.origin !== new URL(url).origin
  } catch {
    return true
  }
}

const getDefaultFileName = (url: string): string => {
  if (!url) return `media-${Date.now()}`

  const pureUrl = url.split('?')[0]
  let fileName = pureUrl.split('/').pop() || `media-${Date.now()}`

  const extMap: Record<string, string> = {
    mp4: 'mp4',
    webm: 'webm',
    avi: 'avi',
    mov: 'mov',
    jpg: 'jpg',
    jpeg: 'jpeg',
    png: 'png',
    webp: 'webp',
    gif: 'gif',
  }

  const hasExt = Object.keys(extMap).some((ext) => fileName.endsWith(`.${ext}`))
  if (!hasExt) {
    for (const [key, ext] of Object.entries(extMap)) {
      if (url.includes(`.${key}`)) {
        fileName = `${fileName}.${ext}`
        break
      }
    }
  }

  return fileName
}

/** Parse filename from Content-Disposition (attachment; filename="..."). */
function parseContentDisposition(header: string | null): string | undefined {
  if (!header) return undefined
  const utf8Match = /filename\*=UTF-8''([^;]+)/i.exec(header)
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1].trim())
    } catch {
      return utf8Match[1].trim()
    }
  }
  const quoted = /filename="([^"]+)"/i.exec(header)
  if (quoted?.[1]) return quoted[1]
  const plain = /filename=([^;]+)/i.exec(header)
  return plain?.[1]?.trim()
}

function triggerBlobDownload(blob: Blob, fileName: string): void {
  const downloadUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = downloadUrl
  link.download = fileName
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(downloadUrl)
}

const downloadSameOrigin = (url: string, fileName: string): void => {
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

async function readDownloadError(res: Response): Promise<string> {
  const text = (await res.text()).trim()
  if (!text) return `HTTP ${res.status}`
  try {
    const body = JSON.parse(text) as { message?: string }
    if (typeof body.message === 'string' && body.message) return body.message
  } catch {
    // plain text error body
  }
  return text
}

/** Gateway proxy for final merged video (SSRF-safe server-side fetch). */
async function downloadWorkplaceFinal(
  sessionKey: string,
  token: string,
  fallbackFileName: string,
): Promise<void> {
  const res = await fetch(
    `/api/workplace/${encodeURIComponent(sessionKey)}/download/final`,
    {
      headers: { Authorization: `Bearer ${token}` },
      credentials: 'same-origin',
    },
  )
  if (!res.ok) {
    throw new Error(await readDownloadError(res))
  }
  const blob = await res.blob()
  const fileName =
    parseContentDisposition(res.headers.get('Content-Disposition')) ||
    fallbackFileName
  triggerBlobDownload(blob, fileName)
}

export const downloadMedia = async (
  options: DownloadMediaOptions,
): Promise<void> => {
  if (isDownloading) {
    console.warn('A download is already in progress.')
    return
  }

  const {
    url,
    fileName: customFileName,
    sessionKey,
    token,
    onError,
  } = options

  const finalFileName = customFileName || getDefaultFileName(url)
  isDownloading = true

  try {
    if (sessionKey && token) {
      await downloadWorkplaceFinal(sessionKey, token, finalFileName)
      return
    }

    if (!url) {
      throw new Error('Download failed: the resource URL is empty.')
    }

    if (isCrossOrigin(url)) {
      console.warn(
        'Cross-origin media downloads require a session key and token, or a same-origin URL.',
      )
      throw new Error('Cross-origin media cannot be downloaded directly.')
    }

    downloadSameOrigin(url, finalFileName)
  } catch (error) {
    const err = error instanceof Error ? error : new Error('Download failed: unknown error.')
    onError?.(err)
    console.error('Media download failed:', err)
    alert(`Download failed: ${err.message}`)
  } finally {
    isDownloading = false
  }
}

export const batchDownloadMedia = async (
  list: DownloadMediaOptions[],
  delay = 300,
): Promise<void> => {
  for (let i = 0; i < list.length; i++) {
    await downloadMedia(list[i])
    if (i < list.length - 1) {
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }
}
