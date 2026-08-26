'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { SystemEvent } from '../schemas/dashboard-schemas';
import { connectEventStream } from '../api/eventsClient';

export function useJobEvents(jobId?: string) {
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const lastEventIdRef = useRef<string | undefined>();

  const handleEvent = useCallback((evt: SystemEvent) => {
    if (!evt) return;
    setEvents((prev) => [...prev, evt]);
    if (evt.event_id) {
      lastEventIdRef.current = evt.event_id;
    }
  }, []);

  useEffect(() => {
    setIsConnected(true);
    const cleanup = connectEventStream({
      jobId,
      lastEventId: lastEventIdRef.current,
      onEvent: handleEvent,
      onError: () => setIsConnected(false)
    });

    return () => {
      setIsConnected(false);
      cleanup();
    };
  }, [jobId, handleEvent]);

  return { events, isConnected, lastEventId: lastEventIdRef.current };
}
