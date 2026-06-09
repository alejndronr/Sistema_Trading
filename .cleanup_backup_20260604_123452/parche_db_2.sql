-- 1. Añadir la columna que faltaba en el journal
ALTER TABLE trades_journal ADD COLUMN IF NOT EXISTS direction VARCHAR(10);

-- 2. Cambiar el DUEÑO de las tablas a 'trading' (esto evita cualquier error de permisos para siempre)
ALTER TABLE manual_investments OWNER TO trading;
ALTER TABLE trades_journal OWNER TO trading;
ALTER TABLE portfolio_state OWNER TO trading;

-- 3. Por si acaso hay secuencias (IDs autoincrementales), también le pasamos la propiedad
DO $$ 
DECLARE 
    r RECORD;
BEGIN 
    FOR r IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public') LOOP 
        EXECUTE 'ALTER SEQUENCE ' || r.sequence_name || ' OWNER TO trading;'; 
    END LOOP; 
END $$;
