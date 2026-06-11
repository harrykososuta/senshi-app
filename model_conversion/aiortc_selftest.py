"""aiortc の ICE 接続が同一プロセス内で成立するかの自己テスト。
ブラウザを介さず、2つの RTCPeerConnection 間で offer/answer + ICE を交換する。
"""
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer


async def main():
    config = RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    pc1 = RTCPeerConnection(config)
    pc2 = RTCPeerConnection(config)

    connected = asyncio.Event()

    @pc1.on("connectionstatechange")
    async def on_state():
        print("pc1 state:", pc1.connectionState)
        if pc1.connectionState == "connected":
            connected.set()

    @pc2.on("connectionstatechange")
    async def on_state2():
        print("pc2 state:", pc2.connectionState)

    # データチャネルを開く（メディアなしでICEだけ検証）
    dc = pc1.createDataChannel("test")

    @dc.on("open")
    def on_open():
        print("DataChannel OPEN -> ICE接続成功!")
        connected.set()

    # offer/answer 手動交換
    await pc1.setLocalDescription(await pc1.createOffer())
    await pc2.setRemoteDescription(pc1.localDescription)
    await pc2.setLocalDescription(await pc2.createAnswer())
    await pc1.setRemoteDescription(pc2.localDescription)

    try:
        await asyncio.wait_for(connected.wait(), timeout=15)
        print("RESULT: OK - aiortcのICE/DTLSはこの環境で正常動作")
    except asyncio.TimeoutError:
        print("RESULT: FAIL - aiortcがローカルでICE接続できない（aiortc/aioice/ファイアウォール問題）")
    finally:
        await pc1.close()
        await pc2.close()


asyncio.run(main())
