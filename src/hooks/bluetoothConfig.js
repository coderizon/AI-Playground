export const DEVICES = [
  {
    id: 'arduino',
    name: 'Arduino UNO R4',
    prefixFilter: ['Arduino UNO R4', 'UNO R4', 'Arduino UNO'],
    serviceId: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
    writeCharCandidates: [
      '6e400002-b5a3-f393-e0a9-e50e24dcca9e',
      '6e400003-b5a3-f393-e0a9-e50e24dcca9e',
    ],
    image: '/assets/images/arduino.png',
  },
  {
    id: 'calliope',
    name: 'Calliope mini',
    prefixFilter: ['Calliope mini', 'CALLIOPE mini', 'CALLIOPE MINI'],
    serviceId: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
    writeCharCandidates: [
      '6e400003-b5a3-f393-e0a9-e50e24dcca9e',
      '6e400002-b5a3-f393-e0a9-e50e24dcca9e',
    ],
    image: '/assets/images/calliope.png',
  },
  {
    id: 'microbit',
    name: 'BBC micro:bit',
    prefixFilter: ['BBC micro:bit', 'micro:bit'],
    serviceId: '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
    writeCharCandidates: [
      '6e400003-b5a3-f393-e0a9-e50e24dcca9e',
      '6e400002-b5a3-f393-e0a9-e50e24dcca9e',
    ],
    image: '/assets/images/microbit.png',
  },
];
