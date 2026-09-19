"""
Script trích xuất toàn diện thông tin thiết bị ONVIF (Device Info, Network, Scopes, Hardware ID).
"""
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

from core.onvif_client import OnvifClient, NS

host = os.getenv("CAMERA_HOST", "192.168.110.14")
port = int(os.getenv("CAMERA_PORT", "80"))
user = os.getenv("CAMERA_USER", "admin")
pwd = os.getenv("CAMERA_PASS", "a12345678")

client = OnvifClient(host, port, user, pwd)
client.discover()

print("--- [1] GetDeviceInformation ---")
try:
    body = '<GetDeviceInformation xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
    root = client._post(client.device_url, body)
    print(ET.tostring(root, encoding="utf-8").decode("utf-8"))
    
    mfg = root.findtext(".//tds:Manufacturer", "", NS)
    model = root.findtext(".//tds:Model", "", NS)
    fw = root.findtext(".//tds:FirmwareVersion", "", NS)
    serial = root.findtext(".//tds:SerialNumber", "", NS)
    hw_id = root.findtext(".//tds:HardwareId", "", NS)
    print(f"Manufacturer:    {mfg}")
    print(f"Model:           {model}")
    print(f"FirmwareVersion: {fw}")
    print(f"SerialNumber:    {serial}")
    print(f"HardwareId:      {hw_id}")
except Exception as e:
    print(f"GetDeviceInformation failed: {e}")

print("\n--- [2] GetNetworkInterfaces (MAC Address) ---")
try:
    body = '<GetNetworkInterfaces xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
    root = client._post(client.device_url, body)
    for iface in root.findall(".//tds:NetworkInterfaces", NS):
        token = iface.get("token", "")
        hw_addr = iface.findtext(".//tt:HwAddress", "", NS)
        ipv4 = iface.findtext(".//tt:IPv4/tt:Config/tt:Manual/tt:Address", "", NS)
        print(f"Interface Token: {token}, MAC: {hw_addr}, IPv4: {ipv4}")
except Exception as e:
    print(f"GetNetworkInterfaces failed: {e}")

print("\n--- [3] GetScopes (Camera Name / Hardware Info) ---")
try:
    body = '<GetScopes xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
    root = client._post(client.device_url, body)
    for sc in root.findall(".//tds:Scopes", NS):
        sc_item = sc.findtext("tds:ScopeItem", "", NS)
        print(f"Scope: {sc_item}")
except Exception as e:
    print(f"GetScopes failed: {e}")

print("\n--- [4] GetHostname ---")
try:
    body = '<GetHostname xmlns="http://www.onvif.org/ver10/device/wsdl"/>'
    root = client._post(client.device_url, body)
    hostname = root.findtext(".//tds:Name", "", NS)
    print(f"Hostname: {hostname}")
except Exception as e:
    print(f"GetHostname failed: {e}")
