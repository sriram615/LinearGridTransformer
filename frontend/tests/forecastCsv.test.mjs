import test from 'node:test'
import assert from 'node:assert/strict'
import { FEATURE_NAMES } from '../src/api/types.ts'
import { CsvValidationError, parseForecastCsv, toForecastRequest } from '../src/api/forecastCsv.ts'

function csv(rows = 144, { headers = [...FEATURE_NAMES], value = '1.25', timestamps = false, timestampAtEnd = false, timestampIndex = 0 } = {}) {
  const resolvedTimestampIndex = timestampAtEnd ? headers.length : timestampIndex
  const columns = timestamps ? [...headers.slice(0, resolvedTimestampIndex), 'DateTime', ...headers.slice(resolvedTimestampIndex)] : headers
  const records = [columns.join(',')]
  for (let index = 0; index < rows; index += 1) {
    const features = headers.map(() => value)
    const date = new Date(Date.UTC(2017, 0, 1, 0, index * 10))
    const stamp = `${date.getUTCMonth() + 1}/${date.getUTCDate()}/${date.getUTCFullYear()} ${date.getUTCHours()}:${String(date.getUTCMinutes()).padStart(2, '0')}`
    const row = [...features]
    if (timestamps) row.splice(resolvedTimestampIndex, 0, stamp)
    records.push(row.join(','))
  }
  return records.join('\n')
}

test('parses canonical 144-row input and builds canonical request', () => {
  const parsed = parseForecastCsv(csv())
  assert.equal(parsed.observations.length, 144)
  assert.deepEqual(parsed.observations[0], Array(8).fill(1.25))
  const request = toForecastRequest(parsed)
  assert.deepEqual(request.feature_names, FEATURE_NAMES)
  assert.equal(request.timestamps, undefined)
  assert.equal(request.observations.length, 144)
})

test('rejects incorrect row count, absent, reordered, and extra feature columns', () => {
  assert.throws(() => parseForecastCsv(csv(143)), /exactly 144/)
  assert.throws(() => parseForecastCsv(csv(144, { headers: FEATURE_NAMES.slice(1) })), /Feature headers/)
  assert.throws(() => parseForecastCsv(csv(144, { headers: [...FEATURE_NAMES].reverse() })), /exact order/)
  assert.throws(() => parseForecastCsv(csv(144, { headers: [...FEATURE_NAMES, 'extra'] })), /exact order/)
})

test('rejects empty files, malformed CSV, missing, non-numeric and non-finite feature values', () => {
  assert.throws(() => parseForecastCsv(''), CsvValidationError)
  assert.throws(() => parseForecastCsv(`${FEATURE_NAMES.join(',')}\n"unterminated`), /unterminated/)
  assert.throws(() => parseForecastCsv(csv(144, { value: '' })), /Missing value/)
  assert.throws(() => parseForecastCsv(csv(144, { value: 'not-a-number' })), /Non-numeric/)
  assert.throws(() => parseForecastCsv(csv(144, { value: 'NaN' })), /Non-numeric/)
  assert.throws(() => parseForecastCsv(csv(144, { value: 'Infinity' })), /Non-numeric/)
  assert.throws(() => parseForecastCsv(csv(144, { value: '1e999' })), /Non-finite/)
})

test('preserves optional DateTime values for backend validation and excludes timestamp from features', () => {
  const parsed = parseForecastCsv(csv(144, { timestamps: true }))
  const request = toForecastRequest(parsed)
  assert.equal(request.timestamps.length, 144)
  assert.equal(request.timestamps[0], '1/1/2017 0:00')
  assert.equal(request.feature_names.length, 8)
  assert.equal(request.observations[0].length, 8)
})

test('supports DateTime as a trailing optional timestamp column without reordering features', () => {
  const parsed = parseForecastCsv(csv(144, { timestamps: true, timestampAtEnd: true }))
  const request = toForecastRequest(parsed)
  assert.equal(request.feature_names.length, 8)
  assert.equal(request.observations[0].length, 8)
  assert.equal(request.timestamps[143], '1/1/2017 23:50')
})

test('supports an optional timestamp between ordered feature columns', () => {
  const parsed = parseForecastCsv(csv(144, { timestamps: true, timestampIndex: 3 }))
  const request = toForecastRequest(parsed)
  assert.deepEqual(request.feature_names, FEATURE_NAMES)
  assert.deepEqual(request.observations[0], Array(8).fill(1.25))
  assert.equal(request.timestamps[0], '1/1/2017 0:00')
})

test('rejects an empty optional timestamp without interpreting its date format', () => {
  const input = csv(144, { timestamps: true }).replace('1/1/2017 0:00,', ',')
  assert.throws(() => parseForecastCsv(input), /Timestamp is missing/)
})
