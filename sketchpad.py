# first inspection, I noticed that albums with id -1 have a lot of missing data
X_train.loc[X_train.album_id == -1, "album_id"] = np.nan
# Question: How many tracks have missing album id?
# Answer: less than 1% of tracks have missing album_id (0.009617730403287856)
sum(np.isnan(X_train.album_id)) / len(X_train.album_id)
# Action: Remove tracks with missing album_id
X_train = X_train[~np.isnan(X_train.album_id)]
# first, we need to dataprocess the datetime to real datetime objects
transformed_dates = pd.to_datetime(X_train['album_date_released'], format = "%Y-%m-%d %H:%M:%S")
# Next, we'll check if every date has transformed rightly
# this function will create a logfile intermediate/dates_log.txt with everything gone wrong
validate_date_transformation(X_train, transformed_dates)
# Result of the validation:
# (1) When a date is present, the date is transformed correctly
# (2) some albums have no album_date_released
# Action: replace the unprocessed dates with the transformed ones
X_train["album_date_released"] = transformed_dates
# Question: Does every track in a given album has the same album_date_released (whether or not it is filled in)
# Answer: yes, missing album_date_released is album specific rather than track specific
X_train.groupby("album_id").agg(
    no_different_values = ("album_date_released", lambda album_dates: len(album_dates.unique()))
).max()
# Question: How many albums have no album_date_released
# Answer: 36% of albums have missing album_date_released
tmp = X_train.groupby("album_id").agg(
    missing = ("album_date_released", lambda album_dates: all(np.isnan(album_dates)))
)
sum(tmp.missing) / len(tmp.missing)
del tmp
# Question: How many tracks have no album_date_released
# Answer 33% of tracks have no album_date_released
sum(np.isnan(X_train.album_date_released)) / len(X_train.album_date_released)
# Note: 36% lays closely to 33%. Which would mean that the number of tracks is roughly equal per album
# Question: Check mean and sd for the number of tracks per album
# Answer: It seems to be the case,
tmp = X_train.groupby("album_id").agg(number_of_tracks=("track_id","count"))
tmp.agg(["count","mean","median","min","max","std"])
mpl.hist(X_train.number_of_tracks[tmp.number_of_tracks<100], bins = 100)