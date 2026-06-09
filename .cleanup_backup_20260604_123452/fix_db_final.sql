-- 1. Crear el usuario 'trading' si no existe, con la contraseña por defecto
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading') THEN
    CREATE ROLE trading WITH LOGIN PASSWORD 'tu_password';
  END IF;
END
$$;

-- Asegurar que la contraseña es correcta y tiene permiso de login
ALTER ROLE trading WITH LOGIN PASSWORD 'tu_password';

-- 2. Hacerle dueño de la base de datos
ALTER DATABASE trading_db OWNER TO trading;

-- 3. Dar permisos absolutos (a 'trading' y, por seguridad local, a 'PUBLIC' para evitar bloqueos de Streamlit)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO PUBLIC;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO PUBLIC;

-- 4. Transferir la propiedad de las tablas
ALTER TABLE IF EXISTS manual_investments OWNER TO trading;
ALTER TABLE IF EXISTS trades_journal OWNER TO trading;
ALTER TABLE IF EXISTS portfolio_state OWNER TO trading;

-- 5. Transferir la propiedad de las secuencias
DO $$ 
DECLARE 
    r RECORD;
BEGIN 
    FOR r IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public') LOOP 
        EXECUTE 'ALTER SEQUENCE ' || r.sequence_name || ' OWNER TO trading;'; 
    END LOOP; 
END $$;
