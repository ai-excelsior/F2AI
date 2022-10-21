import pandas as pd

a = pd.read_parquet("/Users/xuyizhou/Desktop/xyz_warehouse/gitlab/f2ai-credit-scoring/result_table.parquet")
a.to_csv("/Users/xuyizhou/Desktop/xyz_warehouse/gitlab/f2ai-credit-scoring/result_table.csv", index=False)
# Parameter(
#                                     f"row_number() over (partition by {','.join(entity_name)} order by {TIME_COL}_tmp DESC)"
#                                 ),
# --
-- # upload `entity_df` to database
-- to_pgsql(entity_df, TMP_TBL, **self.connection.__dict__)
-- entity_name = entity_df.columns[0]  # entity column name in table
-- views_to_use = {name: view for name, view in feature_view.items() if entity_name in view.entity}
-- sqls = []
-- for view_name, cfg in views_to_use.items():
--     if entity_name in cfg.entity:
--         # time column name in table
--         ent_select = [self.entity[en].entity + " as " + en for en in cfg.entity]
--         fea_select = list(cfg.features.keys())
--         if (
--             not self.sources[cfg.batch_source].event_time
--             and not self.sources[cfg.batch_source].create_time
--         ):  # non time relevant features and is unique for entity
--             sql = f"(SELECT a.{TIME_COL} ,b.* FROM {TMP_TBL} a LEFT JOIN (SELECT {','.join(fea_select + ent_select)} FROM {cfg.batch_source}) b using({entity_name})) as {view_name}"
--         elif not self.sources[cfg.batch_source].create_time:
--             # time relevant features and has no redundency
--             sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] + fea_select )} from \
--                     (SELECT *,row_number() over (partition by c.{entity_name} order by c.{TIME_COL}_b  DESC ) as row_id from \
--                         (SELECT a.{TIME_COL} ,b.* FROM \
--                             {TMP_TBL} a LEFT JOIN \
--                             (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].event_time} as {TIME_COL}_b \
--                             FROM {cfg.batch_source}) b using ({entity_name}) \
--                         where cast(a.{TIME_COL} as date) >= cast(b.{TIME_COL}_b as date)) c \
--                     )tmp where tmp.row_id=1) as {view_name}"
--         elif not self.sources[cfg.batch_source].event_time:
--             # non time relevant features but may have redundency:
--             sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] + fea_select)} from \
--                     (SELECT *,row_number() over (partition by c.{entity_name} order by c.{CREATE_COL}  DESC ) as row_id from \
--                         (SELECT a.{TIME_COL},b.* FROM \
--                             {TMP_TBL} a LEFT JOIN \
--                             (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].create_time} as {CREATE_COL} FROM {cfg.batch_source}) b on using ({entity_name}) ) c \
--                     ) tmp where tmp.row_id=1) as {view_name}"

--         else:
--             sql = f"(SELECT {TIME_COL}, {','.join([en for en in cfg.entity] +fea_select)} from \
--                     (SELECT *,row_number() over (partition by c.{entity_name} order by c.{CREATE_COL} DESC, c.{TIME_COL}_b DESC ) as row_id from \
--                         (SELECT a.{TIME_COL} ,b.* FROM \
--                             {TMP_TBL} a LEFT JOIN \
--                             (SELECT {','.join(fea_select + ent_select)},{self.sources[cfg.batch_source].event_time} as {TIME_COL}_b,{self.sources[cfg.batch_source].create_time} as {CREATE_COL} \
--                             FROM {cfg.batch_source}) b on a.{entity_name}=b.{entity_name} \
--                         where cast(a.{TIME_COL} as date) >= cast(b.{TIME_COL}_b as date)) c \
--                     )tmp where tmp.row_id=1) as {view_name}"
--         sqls.append(sql)
-- final_sql = "SELECT * FROM " + reduce(
--     lambda a, b: f"{a} join {b} using ({entity_name},{TIME_COL})", sqls
-- )
-- result = pd.DataFrame(sql_df(final_sql, conn))
-- result
