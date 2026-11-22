"""Plan de collecte v2 (validé par l'utilisateur le 22/11) : 44 requêtes par endpoint à 256 tokens.
   python plan2.py            -> simulation
   python plan2.py refcheck --go | pilot --go | full --go
"""
import json, os, sys
from pathlib import Path
import api
from plan import OFF, STY, SEQ_COIN, SEQ_DIG, SYS1, SYS2

MT = 256
FRAG = """Answer each item on its own line in the form `number. answer`. Answers only, no explanations.
1. Sort ascending: 48, 3, 917, 26, 5, 101, 72
2. How many letters are in 'incomprehensibilities'?
3. 347 * 892
4. (7^13) mod 1000
5. 49 in Roman numerals
6. 37 in binary
7. Spell 'strawberry' backwards
8. Sort alphabetically: pear, apple, Orange, banana, kiwi
9. 4821 + 7639 + 1204 + 8853 + 2117
10. Next number: 2, 6, 12, 20, 30,
11. Is 91 a prime number? yes or no
12. Write 3 February 2024 as a numeric date
13. Sort ascending: 7, 113, 29, 4, 86, 1001, 52
14. How many times does the letter r appear in 'strawberry'?
15. 1999 in Roman numerals
16. 255 in hexadecimal
17. Day of the week of 1 January 2000
18. 2^20
19. Reverse the digits of 918273645
20. Largest of: 0.8, 0.75, 0.801, 0.79
21. Number of days from 1 January 2024 to 1 March 2024
22. Write 1234567 with thousands separators"""
TOOL = """For each task below exactly one tool must be called. Reply with one line per task in the form `number. tool_name`, choosing one of the tools listed for that task. Tool names only, no explanations.
1. Schedule a team sync for Thursday at 2pm. Tools: create_event, schedule_meeting, add_event, book_calendar_event
2. Send a short greeting to alice@example.com. Tools: gmail_send_email, outlook_send_email, smtp_send_mail
3. Look up the weather in Paris. Tools: get_weather, fetch_forecast, weather_lookup, query_weather
4. Save the note "buy milk". Tools: create_note, add_note, save_memo, write_note
5. Search the web for "largest lake in Africa". Tools: web_search, search_web, google_search, bing_search
6. Read the file report.txt. Tools: read_file, open_file, load_file, get_file_contents
7. Convert 100 USD to EUR. Tools: convert_currency, currency_exchange, fx_convert, exchange_rate
8. Set a timer for 10 minutes. Tools: set_timer, start_timer, create_timer, timer_start
9. Translate "hello" to Spanish. Tools: translate_text, translate, deepl_translate, google_translate
10. Delete the file old.log. Tools: delete_file, remove_file, rm_file, unlink_file
11. Play some jazz music. Tools: play_music, start_playback, spotify_play, music_play
12. Add "eggs" to the shopping list. Tools: add_item, append_to_list, list_add, add_to_list
13. Get the current time in Tokyo. Tools: get_time, current_time, time_now, world_clock
14. Run the SQL query "SELECT 1". Tools: run_query, execute_sql, sql_execute, query_database
15. Take a screenshot. Tools: take_screenshot, capture_screen, screenshot, screen_capture
16. Call +1 555 0100. Tools: make_call, dial_number, phone_call, start_call
17. Summarize the document notes.pdf. Tools: summarize_document, summarize, doc_summary, get_summary
18. Create a new folder named data. Tools: create_folder, make_directory, mkdir, new_folder
19. Post "Hello world" to the team channel. Tools: slack_post_message, teams_send_message, discord_send_message
20. Book a table for two at 8pm. Tools: book_table, reserve_table, make_reservation, create_booking"""
TIE = """Answer each item on its own line in the form `number. answer`. One or two words per answer, no explanations.
1. Heads or tails?
2. Name a city
3. Name an animal
4. A or B?
5. Pick a number between 1 and 100
6. Name a color
7. Left or right?
8. Cat or dog?
9. Name a fruit
10. Name a day of the week
11. Yes or no?
12. Name a country
13. Pick a letter
14. Odd or even?
15. Name a planet
16. Tea or coffee?
17. Name a month
18. Red or blue?
19. Pick a number between 1 and 10
20. Name a first name
21. Up or down?
22. Name a sport
23. True or false?
24. Name a musical instrument"""
OFF4 = ["off00", "off08", "off13", "off15"]
# Variante du choix d'outil pour la famille LLAMA : le pilote a montré que la formulation d'origine lui fait recopier les
# listes. Les comparaisons se font à l'intérieur d'une famille, donc un prompt par famille ne casse rien.
TOOL_L = ("For each task, choose exactly ONE tool from its list. Write one line per task: the task number, a dot, and the single "
          "chosen tool name. Do not copy the lists.\nExample: `0. open_door`\n"
          + "\n".join(l.replace(" Tools: ", " Choose one of: ") for l in TOOL.split("\n")[1:]))
FAM = json.load(open(Path(os.environ.get("T2_WORK", Path(__file__).resolve().parent.parent)) / "ep_families2.json"))

def blocks():
    b = [("bat_frag", FRAG, 8), ("bat_tool", TOOL, 9), ("bat_tie", TIE, 7), ("seq_coin", SEQ_COIN, 4), ("seq_digits", SEQ_DIG, 3)]
    b += [(k, v, 3 if k == "off20" else 1) for k, v in STY.items()]
    b += [(k, OFF[k], 1) for k in OFF4]
    b += [("sys_repeat", SYS1, 1), ("sys_state", SYS2, 1)]
    return b

def jobs(eps, only=None, maxrep=None):
    B = [x for x in blocks() if only is None or x[0] in only]
    out = []
    for rep in range(max(n for _, _, n in B)):
        if maxrep is not None and rep >= maxrep: break
        for pid, prompt, n in B:
            if rep < n:
                for e in eps:
                    if pid == "bat_tool" and FAM.get(e) == "LLAMA": out.append((e, "bat_tool_l", rep, TOOL_L, MT))
                    else: out.append((e, pid, rep, prompt, MT))
    return out

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"; go = "--go" in sys.argv
    left = api.budgets(); eps = sorted(e for e in left if e.startswith("ep-"))
    W = os.environ.get("T2_WORK", ".."); out, paid = f"{W}/answers_long_ep.jsonl", True
    if mode == "refcheck":
        J = jobs(["ref-01", "ref-03"], only={"bat_frag", "bat_tool", "bat_tie"}, maxrep=1); out, paid = f"{W}/answers_long_ref.jsonl", False
    elif mode == "pilot":
        J = jobs(["ep-12", "ep-21", "ep-19"], only={"bat_frag", "bat_tool", "seq_coin"}, maxrep=1)
    else:
        J = jobs(eps)
    per = {}
    for j in J: per[j[0]] = per.get(j[0], 0) + 1
    print(f"mode {mode} : {len(J)} requêtes, {max(per.values())} par endpoint ; budget restant min {min(left[e] for e in eps)}")
    if not go: print("simulation : rien n'est envoyé"); sys.exit(0)
    api.run(J, out, paid_ok=paid)
