'use client';

import { useEffect, useState, useCallback } from 'react';
import { SystemEvent } from '../schemas/dashboard-schemas';
import { connectEventStream } from '../api/eventsClient';

export function useJobEvents(jobId?: string) {
  const [events, setEvents] = useState<SystemEvent[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastEventId, setLastEventId] = useState<string | undefined>();

  const handleEvent = useCallback((evt: SystemEvent) => {
    setEvents((prev) => [...prev, evt]);
    setLastEventId(evt.event_id);
  }, []);

  useEffect(() => {
    setIsConnected(true);
    const cleanup = connectEventStream({
      jobId,
      lastEventId,
      onEvent: handleEvent,
      onError: () => setIsConnected(false)
    });

    return () => {
      setIsConnected(false);
      cleanup();
    };
  }, [jobId, handleEvent, lastEventId]);

  return { events, isConnected, lastEventId };
}
