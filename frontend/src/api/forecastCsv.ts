import { FEATURE_NAMES, type ForecastRequest } from './types.ts'

export interface ParsedForecastCsv {
  headers: string[]
  observations: number[][]
  timestamps?: string[]
}

export class CsvValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CsvValidationError'
  }
}

function parseRecords(text: string): string[][] {
  if (!text.trim()) throw new CsvValidationError('The selected CSV is empty.')
  const records: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  let afterQuote = false

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i]
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i += 1
        } else {
          quoted = false
          afterQuote = true
        }
      } else {
        field += char
      }
      continue
    }

    if (afterQuote && char !== ',' && char !== '\n' && char !== '\r') {
      throw new CsvValidationError(`Unexpected character after a quoted field at character ${i + 1}.`)
    }
    if (char === '"') {
      if (field.length !== 0 || afterQuote) {
        throw new CsvValidationError(`Unexpected quote at character ${i + 1}.`)
      }
      quoted = true
    } else if (char === ',') {
      row.push(field)
      field = ''
      afterQuote = false
    } else if (char === '\n' || char === '\r') {
      row.push(field)
      records.push(row)
      row = []
      field = ''
      afterQuote = false
      if (char === '\r' && text[i + 1] === '\n') i += 1
    } else {
      field += char
    }
  }

  if (quoted) throw new CsvValidationError('The CSV contains an unterminated quoted field.')
  if (row.length > 0 || field.length > 0 || afterQuote) {
    row.push(field)
    records.push(row)
  }
  return records
}

export function parseForecastCsv(text: string): ParsedForecastCsv {
  const records = parseRecords(text)
  if (records.length === 0) throw new CsvValidationError('The CSV contains a header but no observations.')

  const headers = records[0].map((header, index) => index === 0 ? header.replace(/^\uFEFF/, '') : header)
  const timestampIndices = headers.flatMap((header, index) => header === 'timestamp' || header === 'DateTime' ? [index] : [])
  if (timestampIndices.length > 1) {
    throw new CsvValidationError('Use at most one optional timestamp column.')
  }
  const timestampIndex = timestampIndices[0] ?? -1
  const featureHeaders = headers.filter((_, index) => index !== timestampIndex)
  if (featureHeaders.length !== FEATURE_NAMES.length || featureHeaders.some((header, index) => header !== FEATURE_NAMES[index])) {
    throw new CsvValidationError(`Feature headers must match this exact order: ${FEATURE_NAMES.join(', ')}. One optional timestamp or DateTime column may appear anywhere without changing feature order.`)
  }
  if (headers.length !== FEATURE_NAMES.length + (timestampIndex >= 0 ? 1 : 0)) {
    throw new CsvValidationError('The CSV must contain exactly eight feature columns and at most one timestamp column.')
  }

  const dataRows = records.slice(1)
  if (dataRows.length !== 144) {
    throw new CsvValidationError(`Expected exactly 144 observations; found ${dataRows.length}.`)
  }

  const featureIndices = headers.map((_, index) => index).filter((index) => index !== timestampIndex)
  const observations: number[][] = []
  const timestamps: string[] = []
  for (let rowIndex = 0; rowIndex < dataRows.length; rowIndex += 1) {
    const row = dataRows[rowIndex]
    if (row.length !== headers.length) {
      throw new CsvValidationError(`Observation ${rowIndex + 1} has ${row.length} fields; expected ${headers.length}.`)
    }
    if (timestampIndex >= 0) {
      const timestamp = row[timestampIndex]
      if (!timestamp.trim()) throw new CsvValidationError(`Timestamp is missing on observation ${rowIndex + 1}.`)
      timestamps.push(timestamp)
    }
    const values = featureIndices.map((columnIndex) => {
      const raw = row[columnIndex].trim()
      if (!raw) throw new CsvValidationError(`Missing value in ${headers[columnIndex]} on observation ${rowIndex + 1}.`)
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(raw)) {
        throw new CsvValidationError(`Non-numeric value in ${headers[columnIndex]} on observation ${rowIndex + 1}.`)
      }
      const value = Number(raw)
      if (!Number.isFinite(value)) throw new CsvValidationError(`Non-finite value in ${headers[columnIndex]} on observation ${rowIndex + 1}.`)
      return value
    })
    observations.push(values)
  }

  return {
    headers,
    observations,
    ...(timestampIndex >= 0 ? { timestamps } : {}),
  }
}

export function toForecastRequest(parsed: ParsedForecastCsv): ForecastRequest {
  return {
    feature_names: [...FEATURE_NAMES],
    observations: parsed.observations,
    ...(parsed.timestamps ? { timestamps: [...parsed.timestamps] } : {}),
  }
}
