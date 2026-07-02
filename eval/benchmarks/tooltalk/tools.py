"""
Static OpenAI-style tool schemas for all 28 ToolTalk tools.

Extracted from microsoft/ToolTalk source code to avoid runtime
dependency on sent2vec.  Organized by the 7 suites.
"""

def _t(name, desc, params, required):
    """Helper to build an OpenAI-style function tool dict."""
    props = {}
    for pname, ptype, pdesc in params:
        props[pname] = {"type": ptype, "description": pdesc}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props},
            "required": required,
        },
    }


SUITES = {
    "AccountTools": [
        _t("GetAccountInformation", "Retrieves account information of logged in user.", [], []),
        _t("DeleteAccount", "Deletes a user's account, requires user to be logged in.",
           [("password", "string", "The password of the user.")], ["password"]),
        _t("UserLogin", "Logs in a user returns a token.",
           [("username", "string", "The username of the user."),
            ("password", "string", "The password of the user.")], ["username", "password"]),
        _t("LogoutUser", "Logs user out.", [], []),
        _t("ChangePassword", "Changes the password of an account.",
           [("old_password", "string", "The old password of the user."),
            ("new_password", "string", "The new password of the user.")], ["old_password", "new_password"]),
        _t("RegisterUser", "Register a new user.",
           [("username", "string", "The username of the user."),
            ("password", "string", "The password of the user."),
            ("email", "string", "The email of the user."),
            ("name", "string", "The name of the user."),
            ("phone", "string", "The phone of the user in the format xxx-xxx-xxxx.")],
           ["username", "password", "email"]),
        _t("SendVerificationCode", "Initiates a password reset for a user by sending a verification code to a backup email.",
           [("username", "string", "The username of the user."),
            ("email", "string", "The email of the user.")], ["username", "email"]),
        _t("ResetPassword", "Resets the password of a user using a verification code.",
           [("username", "string", "The username of the user."),
            ("verification_code", "string", "The 6 digit verification code sent to the user."),
            ("new_password", "string", "The new password of the user.")],
           ["username", "verification_code", "new_password"]),
        _t("QueryUser", "Finds users given a username or email.",
           [("username", "string", "The username of the user, required if email is not supplied."),
            ("email", "string", "The email of the user, required if username is not supplied.")], []),
        _t("UpdateAccountInformation", "Updates account information of a user.",
           [("password", "string", "The password of the user."),
            ("new_email", "string", "The new email of the user."),
            ("new_phone_number", "string", "The new phone number of the user in the format xxx-xxx-xxxx."),
            ("new_name", "string", "The new name of the user.")], ["password"]),
    ],

    "Alarm": [
        _t("AddAlarm", "Adds an alarm for a set time.",
           [("time", "string", "The time for alarm. Format: %H:%M:%S")], ["time"]),
        _t("DeleteAlarm", "Deletes an alarm given an alarm_id.",
           [("alarm_id", "string", "Alarm ID. Format: xxxx-xxxx.")], ["alarm_id"]),
        _t("FindAlarms", "Finds alarms the user has set.",
           [("start_range", "string", "Optional starting time range to find alarms. Format: %H:%M:%S"),
            ("end_range", "string", "Optional ending time range to find alarms. Format: %H:%M:%S")], []),
    ],

    "Calendar": [
        _t("CreateEvent", "Adds events to a user's calendar.",
           [("name", "string", "The name of the event."),
            ("event_type", "string", "The type of the event, either 'meeting' or 'event'."),
            ("description", "string", "The description of the event, no more than 1024 characters."),
            ("start_time", "string", "The start time of the event, in the pattern of %Y-%m-%d %H:%M:%S"),
            ("end_time", "string", "The end time of the event, in the pattern of %Y-%m-%d %H:%M:%S."),
            ("location", "string", "Optional, the location where the event is to be held."),
            ("attendees", "array", "The attendees as an array of usernames. Required if event type is meeting.")],
           ["name", "event_type", "start_time", "end_time"]),
        _t("DeleteEvent", "Deletes events from a user's calendar.",
           [("event_id", "string", "The id of the event to be deleted.")], ["event_id"]),
        _t("ModifyEvent", "Allows modification of an existing event.",
           [("event_id", "string", "The id of the event to be modified."),
            ("new_name", "string", "The new name of the event."),
            ("new_start_time", "string", "The new start time of the event."),
            ("new_end_time", "string", "The new end time of the event. Required if new_start_time is provided."),
            ("new_description", "string", "The new description of the event."),
            ("new_location", "string", "The new location of the event."),
            ("new_attendees", "array", "The new attendees of the event. Array of usernames.")],
           ["event_id"]),
        _t("QueryCalendar", "Query for events that occur in a time range.",
           [("start_time", "string", "The start time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S"),
            ("end_time", "string", "The end time of the meeting, in the pattern of %Y-%m-%d %H:%M:%S")],
           ["start_time", "end_time"]),
    ],

    "Email": [
        _t("SendEmail", "Sends an email on behalf of a given user.",
           [("to", "array", "Receiving addresses of the email."),
            ("subject", "string", "The subject of the email."),
            ("body", "string", "The content of the email.")], ["to", "subject", "body"]),
        _t("SearchInbox", "Searches for emails matching filters returning 5 most recent results.",
           [("query", "string", "Query containing keywords to search for."),
            ("match_type", "string", "Whether to match any or all keywords. Defaults to any."),
            ("sender", "string", "The email address of the sender."),
            ("start_date", "string", "Starting time to search for, in the pattern of %Y-%m-%d %H:%M:%S."),
            ("end_date", "string", "End time to search for, in the pattern of %Y-%m-%d %H:%M:%S.")], []),
    ],

    "Messages": [
        _t("SendMessage", "Sends a message to another user.",
           [("receiver", "string", "The receiver's username."),
            ("message", "string", "The message.")], ["receiver", "message"]),
        _t("SearchMessages", "Searches messages matching filters returning 5 most recent results.",
           [("query", "string", "Query containing keywords to search for."),
            ("match_type", "string", "Whether to match any or all keywords. Defaults to any."),
            ("sender", "string", "Username of the sender."),
            ("start_date", "string", "Starting time to search for, in the pattern of %Y-%m-%d %H:%M:%S."),
            ("end_date", "string", "End time to search for, in the pattern of %Y-%m-%d %H:%M:%S.")], []),
    ],

    "Reminder": [
        _t("AddReminder", "Add a reminder.",
           [("task", "string", "The task to be reminded of."),
            ("due_date", "string", "Optional date the task is due, in the format of %Y-%m-%d %H:%M:%S.")],
           ["task"]),
        _t("GetReminders", "Get a list of reminders.", [], []),
        _t("DeleteReminder", "Delete a reminder.",
           [("reminder_id", "string", "The reminder_id of the reminder to be deleted.")], ["reminder_id"]),
        _t("CompleteReminder", "Complete a reminder.",
           [("reminder_id", "string", "The reminder_id of the reminder to be deleted.")], ["reminder_id"]),
    ],

    "Weather": [
        _t("CurrentWeather", "Get the current weather of a location.",
           [("location", "string", "The location to get the weather of.")], ["location"]),
        _t("ForecastWeather", "Get the 3-day forecast weather of a location.",
           [("location", "string", "The location to get the weather of.")], ["location"]),
        _t("HistoricWeather", "Get historic weather information of a location by month.",
           [("location", "string", "The location to get the weather of."),
            ("month", "string", "The month to get weather of as a full name.")], ["location", "month"]),
    ],
}
