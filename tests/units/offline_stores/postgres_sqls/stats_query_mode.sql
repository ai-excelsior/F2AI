SELECT "zipcode",MODE() WITHIN GROUP (ORDER BY population) as population FROM "zipcode_table" GROUP BY "zipcode"