ALTER TABLE jobs ADD COLUMN extraction_result TEXT;
ALTER TABLE jobs ADD COLUMN last_error_kind TEXT;

UPDATE jobs
SET extraction_result = substr(last_error, 8),
    last_error = NULL
WHERE extraction_result IS NULL
  AND last_error LIKE 'result:%';
