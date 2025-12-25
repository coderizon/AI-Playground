import { useCallback, useRef, useState } from 'react';

export function useBluetooth() {
  const [isConnected, setIsConnected] = useState(false);
  const [device, setDevice] = useState(null);
  const characteristicRef = useRef(null);
  const deviceRef = useRef(null);

  const resetConnection = useCallback(() => {
    characteristicRef.current = null;
    deviceRef.current = null;
    setIsConnected(false);
    setDevice(null);
  }, []);

  const connect = useCallback(
    async (deviceConfig) => {
      try {
        if (!navigator?.bluetooth) {
          throw new Error('Web Bluetooth wird von diesem Browser nicht unterstützt.');
        }

        const filters = deviceConfig.prefixFilter.map((prefix) => ({ namePrefix: prefix }));
        filters.push({ services: [deviceConfig.serviceId] });

        const nextDevice = await navigator.bluetooth.requestDevice({
          filters,
          optionalServices: [deviceConfig.serviceId],
        });

        if (!nextDevice.gatt) {
          throw new Error('Kein kompatibler GATT-Server gefunden.');
        }

        const server = await nextDevice.gatt.connect();
        const service = await server.getPrimaryService(deviceConfig.serviceId);

        let characteristic = null;
        for (const uuid of deviceConfig.writeCharCandidates) {
          try {
            characteristic = await service.getCharacteristic(uuid);
            if (characteristic) break;
          } catch (error) {
            continue;
          }
        }

        if (!characteristic) {
          throw new Error('Kein kompatibles UART-Charakteristikum gefunden.');
        }

        deviceRef.current = nextDevice;
        characteristicRef.current = characteristic;

        nextDevice.addEventListener('gattserverdisconnected', () => {
          resetConnection();
        });

        setDevice(deviceConfig);
        setIsConnected(true);
      } catch (error) {
        console.error(error);
        resetConnection();
        alert(`Fehler: ${error.message}`);
      }
    },
    [resetConnection],
  );

  const send = useCallback(async (data) => {
    if (!characteristicRef.current) return;

    try {
      const encoded = new TextEncoder().encode(`${data}\n`);
      if (characteristicRef.current.writeValueWithoutResponse) {
        await characteristicRef.current.writeValueWithoutResponse(encoded);
      } else {
        await characteristicRef.current.writeValue(encoded);
      }
    } catch (error) {
      console.error('Send failed', error);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (deviceRef.current?.gatt?.connected) {
      deviceRef.current.gatt.disconnect();
    }
    resetConnection();
  }, [resetConnection]);

  return { connect, disconnect, send, isConnected, device };
}
