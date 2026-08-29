import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";

import { ThreadComposer } from "@/components/thread/ThreadComposer";
import { ThreadHeader } from "@/components/thread/ThreadHeader";
import { StreamErrorNotice } from "@/components/thread/StreamErrorNotice";
import { ThreadViewport } from "@/components/thread/ThreadViewport";
import {
  type ThreadStreamControl,
  useThreadStream,
} from "@/hooks/useThreadStream";
import type { SendImage } from "@/hooks/useNanobotStream";
import {
  useFirstFrameImage,
  type FirstFrameData,
} from "@/hooks/useFirstFrameImage";
import { useClipboardAndDrop } from "@/hooks/useClipboardAndDrop";
import { resolveQuestionCardsFromThread } from "@/lib/questions";
import {
  ApiError,
  deleteReferenceImage,
  fetchGenerationSettings,
  saveGenerationSettings,
  uploadFirstFrameImage,
} from "@/lib/api";
import { classifyFirstFrameQuestion } from "@/lib/first-frame-question-actions";
import type {
  ChatSummary,
  MemoryReview,
  MemoryWorkspaceAsset,
  ShotMemoryAssetCreate,
  UIMessage,
  WorkplaceReferenceImage,
} from "@/lib/types";
import type { NextContinuousControl } from "@/components/thread/MemoryReviewCard";
import { useClient } from "@/providers/ClientProvider";
import {
  LONG_VIDEO_DEFAULT_DURATION,
  LONG_VIDEO_DURATION_OPTIONS,
} from "@/components/thread/DurationPicker";
import {
  DEFAULT_VIDEO_SIZE,
  type VideoSize,
} from "@/components/thread/AspectRatioPicker";
import {
  DEFAULT_STORY_LANGUAGE,
  type StoryLanguage,
  STORY_LANGUAGE_PRESETS,
} from "@/components/thread/LanguagePicker";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { toast } from "@/components/ui/Toast";
import { FIRST_FRAME_UPLOAD_ENABLED } from "@/config/features";
import { cn } from "@/lib/utils";

function parseStoryLanguage(value: unknown): StoryLanguage {
  if (typeof value !== "string") return DEFAULT_STORY_LANGUAGE;
  const cleaned = value.trim();
  if ((STORY_LANGUAGE_PRESETS as readonly string[]).includes(cleaned)) {
    return cleaned as StoryLanguage;
  }
  const lowered = cleaned.toLowerCase();
  if (
    lowered === "zh" ||
    lowered === "zh-cn" ||
    lowered === "chinese" ||
    cleaned === "中文"
  ) {
    return "zh";
  }
  if (lowered === "en" || lowered === "en-us" || lowered === "english") {
    return "en";
  }
  return DEFAULT_STORY_LANGUAGE;
}

interface ThreadShellProps {
  session: ChatSummary | null;
  title: string;
  onToggleSidebar: () => void;
  onGoHome: () => void;
  onNewChat: () => Promise<string | null>;
  hideSidebarToggleOnDesktop?: boolean;
  /** Same as sidebar refresh — e.g. reload session titles after a reply. */
  onReplyEnd?: () => void;
  /** When provided, stream state is owned by the parent (e.g. ThreadWorkplaceShell). */
  stream?: ThreadStreamControl;
  mockMessages?: UIMessage[];
  mockStreaming?: boolean;
  mockSend?: (content: string) => void;
  actionSlot?: ReactNode;
  modeSelectorSlot?: ReactNode;
  onMessageComplete?: (message: UIMessage) => void;
  onAnswerQuestion?: (messageId: string, cardId: string, value: string) => void;
  onMemoryReviewAction?: (
    review: MemoryReview,
    action: "approve" | "reselect" | "manual_select" | "select_mode",
    memoryId?: string,
    timestampSec?: number,
    selectionMode?: "manual" | "vlm",
    retainedMemoryIds?: string[],
  ) => void | Promise<void>;
  memoryAssets?: MemoryWorkspaceAsset[];
  onCreateMemoryAsset?: (
    shotId: number,
    asset: ShotMemoryAssetCreate,
  ) => Promise<void>;
  getNextContinuous?: (
    review: MemoryReview,
  ) => NextContinuousControl | null;
  /** Blocks chat send while the right-side workplace textarea is focused. */
  composerSendDisabled?: boolean;
  videoSizeLocked?: boolean;
  referenceImageLocked?: boolean;
  persistedReferenceImage?: WorkplaceReferenceImage | null;
  persistedAutoGenerate?: boolean;
  forceAutoGenerate?: boolean;
  onReferenceImageSynced?: () => void;
}

function toModelBadgeLabel(modelName: string | null): string | null {
  if (!modelName) return null;
  const trimmed = modelName.trim();
  if (!trimmed) return null;
  const leaf = trimmed.split("/").pop() ?? trimmed;
  return leaf || trimmed;
}

type ThreadShellInnerProps = Omit<ThreadShellProps, "stream" | "onReplyEnd"> & {
  stream: ThreadStreamControl;
};

function ThreadShellInner({
  session,
  title,
  onToggleSidebar,
  onGoHome,
  onNewChat,
  hideSidebarToggleOnDesktop = false,
  stream,
  mockMessages,
  mockStreaming,
  mockSend,
  actionSlot,
  modeSelectorSlot,
  onMessageComplete,
  onAnswerQuestion,
  onMemoryReviewAction,
  memoryAssets,
  onCreateMemoryAsset,
  getNextContinuous,
  composerSendDisabled = false,
  videoSizeLocked = false,
  referenceImageLocked = false,
  persistedReferenceImage = null,
  persistedAutoGenerate = false,
  forceAutoGenerate = false,
  onReferenceImageSynced,
}: ThreadShellInnerProps) {
  const { t } = useTranslation();
  const {
    chatId,
    loading,
    messages,
    isStreaming,
    turnComplete,
    send,
    answerQuestion,
    streamError,
    dismissStreamError,
  } = stream;
  const { client, modelName, token } = useClient();
  const [booting, setBooting] = useState(false);
  const [durationSec, setDurationSec] = useState(LONG_VIDEO_DEFAULT_DURATION);
  const [videoSize, setVideoSize] = useState<VideoSize>(DEFAULT_VIDEO_SIZE);
  const [storyLanguage, setStoryLanguage] = useState<StoryLanguage>(
    DEFAULT_STORY_LANGUAGE,
  );
  const [autoGenerate, setAutoGenerate] = useState(
    forceAutoGenerate || persistedAutoGenerate,
  );
  const [firstFrameUploadFailedOpen, setFirstFrameUploadFailedOpen] =
    useState(false);
  const [firstFrameUploadError, setFirstFrameUploadError] = useState("");
  const [referencePreviewDismissed, setReferencePreviewDismissed] =
    useState(false);
  const durationSecRef = useRef(LONG_VIDEO_DEFAULT_DURATION);
  const videoSizeRef = useRef<VideoSize>(DEFAULT_VIDEO_SIZE);
  const storyLanguageRef = useRef<StoryLanguage>(DEFAULT_STORY_LANGUAGE);
  const pendingDurationSaveRef = useRef(false);
  const pendingFirstRef = useRef<string | null>(null);
  const uploadedRefImageRef = useRef<WorkplaceReferenceImage | null>(null);
  const lastPersistKeyRef = useRef<string | null>(null);
  const autoGenerateRef = useRef(autoGenerate);
  autoGenerateRef.current = autoGenerate;
  const showHeroComposer = messages.length === 0 && !loading;

  durationSecRef.current = durationSec;
  videoSizeRef.current = videoSize;
  storyLanguageRef.current = storyLanguage;

  const isMockMode = !!mockSend;
  const activeMessages = isMockMode
    ? (mockMessages ?? [])
    : resolveQuestionCardsFromThread(messages);
  const activeStreaming = isMockMode ? (mockStreaming ?? false) : isStreaming;
  const activeTurnComplete = isMockMode ? true : turnComplete;
  const activeSend = isMockMode ? mockSend! : send;
  const showHero = isMockMode ? activeMessages.length === 0 : showHeroComposer;

  const showFirstFrameUploader = !isMockMode;

  const onFirstFrameReject = useCallback(
    (reason: "unsupported_type" | "decode_failed") => {
      toast.error(t(`thread.firstFrame.rejected.${reason}`));
    },
    [t],
  );

  const {
    value: firstFrame,
    cropping: firstFrameCropping,
    setFile: setFirstFrameFile,
    clear: clearFirstFrame,
  } = useFirstFrameImage(videoSize, onFirstFrameReject);

  const firstFrameValueRef = useRef(firstFrame);
  firstFrameValueRef.current = firstFrame;

  const onPageImageFiles = useCallback(
    (files: File[]) => {
      if (
        !FIRST_FRAME_UPLOAD_ENABLED ||
        !showFirstFrameUploader ||
        files.length === 0
      ) {
        return;
      }
      void setFirstFrameFile(files[0]!);
    },
    [showFirstFrameUploader, setFirstFrameFile],
  );

  const {
    isDragging,
    onDragEnter,
    onDragOver,
    onDragLeave,
    onDrop,
  } = useClipboardAndDrop(onPageImageFiles);

  const prepareStepwiseReference = useCallback(
    async (frame: FirstFrameData): Promise<WorkplaceReferenceImage | null> => {
      const uploaded = await uploadFirstFrameImage(frame.croppedBlob, frame.name);
      const image: WorkplaceReferenceImage = {
        url: uploaded.url,
        name: frame.name,
        width: uploaded.width || frame.width,
        height: uploaded.height || frame.height,
      };
      uploadedRefImageRef.current = image;
      return image;
    },
    [],
  );

  const persistStepwiseReference = useCallback(
    async (frame: FirstFrameData): Promise<WorkplaceReferenceImage | null> => {
      const image = await prepareStepwiseReference(frame);
      if (!image || !session?.key || !chatId) return image;
      try {
        await client.saveWorkplaceReferenceImage(chatId, image);
        lastPersistKeyRef.current = `${frame.name}:${frame.width}x${frame.height}:${frame.croppedBlob.size}`;
        onReferenceImageSynced?.();
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          toast.error(t("thread.firstFrame.locked"));
          return image;
        }
        throw err;
      }
      return image;
    },
    [
      chatId,
      client,
      onReferenceImageSynced,
      prepareStepwiseReference,
      session?.key,
      t,
    ],
  );

  const showFirstFrameError = useCallback((error: unknown) => {
    const detail =
      error instanceof Error && error.message
        ? error.message
        : "unknown first-frame error";
    console.error("First-frame send failed:", error);
    setFirstFrameUploadError(detail);
    setFirstFrameUploadFailedOpen(true);
  }, []);

  const onPickFirstFrame = useCallback(
    (file: File) => {
      if (!FIRST_FRAME_UPLOAD_ENABLED) return;
      if (referenceImageLocked) {
        toast.error(t("thread.firstFrame.locked"));
        return;
      }
      setReferencePreviewDismissed(false);
      void setFirstFrameFile(file);
    },
    [referenceImageLocked, setFirstFrameFile, t],
  );

  const onClearFirstFrame = useCallback(() => {
    if (referenceImageLocked) {
      toast.error(t("thread.firstFrame.locked"));
      return;
    }
    clearFirstFrame();
    setReferencePreviewDismissed(true);
    uploadedRefImageRef.current = null;
    if (session?.key) {
      void deleteReferenceImage(token, session.key)
        .then(() => onReferenceImageSynced?.())
        .catch((err) => {
          if (err instanceof ApiError && err.status === 409) {
            toast.error(t("thread.firstFrame.locked"));
          }
        });
    }
  }, [
    clearFirstFrame,
    onReferenceImageSynced,
    referenceImageLocked,
    session?.key,
    t,
    token,
  ]);

  const onFirstFrameLockedAttempt = useCallback(() => {
    toast.error(t("thread.firstFrame.locked"));
  }, [t]);

  useEffect(() => {
    setAutoGenerate(Boolean(forceAutoGenerate || persistedAutoGenerate));
  }, [session?.key, forceAutoGenerate, persistedAutoGenerate]);

  useEffect(() => {
    if (firstFrame || !persistedReferenceImage?.url) {
      return;
    }
    uploadedRefImageRef.current = persistedReferenceImage;
  }, [firstFrame, persistedReferenceImage]);

  const prevStreamingIdRef = useRef<string | null>(null);

  useEffect(() => {
    lastPersistKeyRef.current = null;
    setReferencePreviewDismissed(false);
    clearFirstFrame();
  }, [session?.key, clearFirstFrame]);

  useEffect(() => {
    if (isMockMode || !session?.key) return;
    let cancelled = false;
    void (async () => {
      try {
        if (pendingDurationSaveRef.current) {
          await saveGenerationSettings(
            token,
            session.key,
            durationSecRef.current,
            videoSizeRef.current.width,
            videoSizeRef.current.height,
            storyLanguageRef.current,
          );
          pendingDurationSaveRef.current = false;
          return;
        }
        const settings = await fetchGenerationSettings(token, session.key);
        if (!cancelled) {
          setDurationSec(settings.duration_sec);
          setVideoSize({ width: settings.width, height: settings.height });
          setStoryLanguage(parseStoryLanguage(settings.language));
        }
      } catch {
        // keep local value on fetch/save failure
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session?.key, isMockMode, token]);

  const onDurationChange = useCallback(
    (sec: number) => {
      setDurationSec(sec);
      if (isMockMode) return;
      if (session?.key) {
        void saveGenerationSettings(token, session.key, sec).catch(() => {});
      } else {
        pendingDurationSaveRef.current = true;
      }
    },
    [isMockMode, session?.key, token],
  );

  const onVideoSizeChange = useCallback(
    (size: VideoSize) => {
      setVideoSize(size);
      if (isMockMode || videoSizeLocked) return;
      if (session?.key) {
        void saveGenerationSettings(
          token,
          session.key,
          durationSecRef.current,
          size.width,
          size.height,
          storyLanguageRef.current,
        ).catch(() => {});
      } else {
        pendingDurationSaveRef.current = true;
      }
    },
    [isMockMode, session?.key, token, videoSizeLocked],
  );

  const onStoryLanguageChange = useCallback(
    (language: StoryLanguage) => {
      setStoryLanguage(language);
      if (isMockMode) return;
      if (session?.key) {
        void saveGenerationSettings(
          token,
          session.key,
          durationSecRef.current,
          videoSizeRef.current.width,
          videoSizeRef.current.height,
          language,
        ).catch(() => {});
      } else {
        pendingDurationSaveRef.current = true;
      }
    },
    [isMockMode, session?.key, token],
  );

  useEffect(() => {
    if (!onMessageComplete) return;
    const last = activeMessages[activeMessages.length - 1];
    if (last?.role === "assistant" && last.isStreaming) {
      prevStreamingIdRef.current = last.id;
    } else if (
      prevStreamingIdRef.current &&
      last?.role === "assistant" &&
      !last.isStreaming &&
      last.id === prevStreamingIdRef.current
    ) {
      prevStreamingIdRef.current = null;
      onMessageComplete(last);
    }
  }, [activeMessages, onMessageComplete]);

  useEffect(() => {
    if (!chatId) return;
    const pending = pendingFirstRef.current;
    if (!pending) return;
    pendingFirstRef.current = null;

    const ref = uploadedRefImageRef.current;
    send(pending, undefined, {
      autoGenerate: autoGenerateRef.current,
      duration_sec: durationSecRef.current,
      ...(ref?.url
        ? {
            reference_image_url: ref.url,
            reference_image_name: ref.name,
            reference_image_width: ref.width,
            reference_image_height: ref.height,
          }
        : {}),
    });
    clearFirstFrame();
    setReferencePreviewDismissed(true);
    setBooting(false);
  }, [chatId, clearFirstFrame, send, client]);

  const handleWelcomeSend = useCallback(
    async (content: string) => {
      if (booting) return;
      const frame = firstFrameValueRef.current;
      if (frame) {
        try {
          await prepareStepwiseReference(frame);
        } catch (error) {
          showFirstFrameError(error);
          return false;
        }
      }
      pendingFirstRef.current = content;
      setBooting(true);
      const newId = await onNewChat();
      if (!newId) {
        pendingFirstRef.current = null;
        setBooting(false);
        return false;
      }
      return true;
    },
    [booting, onNewChat, prepareStepwiseReference, showFirstFrameError],
  );

  const guardedSessionSend = useCallback(
    async (content: string, images?: SendImage[]) => {
      const frame = firstFrameValueRef.current;
      let ref = uploadedRefImageRef.current;
      if (frame) {
        try {
          ref = await prepareStepwiseReference(frame);
        } catch (error) {
          showFirstFrameError(error);
          return false;
        }
      }
      const outgoingImages = ref?.url
        ? [
            ...(images ?? []),
            {
              preview: {
                url: ref.url,
                name: ref.name || "first-frame.jpg",
              },
            },
          ]
        : images;
      send(
        content,
        outgoingImages,
        {
          autoGenerate: autoGenerateRef.current,
          duration_sec: durationSecRef.current,
          ...(ref?.url
            ? {
                reference_image_url: ref.url,
                reference_image_name: ref.name,
                reference_image_width: ref.width,
                reference_image_height: ref.height,
              }
            : {}),
        },
      );
      if (frame) {
        clearFirstFrame();
        setReferencePreviewDismissed(true);
      }
      return true;
    },
    [
      clearFirstFrame,
      prepareStepwiseReference,
      send,
      showFirstFrameError,
    ],
  );

  const emptyState = loading ? (
    <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
      {t("thread.loadingConversation")}
    </div>
  ) : modeSelectorSlot ? (
    modeSelectorSlot
  ) : (
    <div className="flex w-full flex-col items-center gap-5 text-center animate-in fade-in-0 slide-in-from-bottom-2 duration-500">
      <h2
        className="text-[43px] font-light tracking-[0.02em] text-foreground/80"
        style={{ fontFamily: "'Cormorant Garamond', Georgia, serif" }}
      >
        Echo Director
      </h2>
    </div>
  );

  const displayFirstFrame =
    firstFrame ??
    (!referencePreviewDismissed && persistedReferenceImage?.url
      ? {
          previewUrl: persistedReferenceImage.url,
          croppedBlob: new Blob(),
          width: persistedReferenceImage.width ?? 0,
          height: persistedReferenceImage.height ?? 0,
          name: persistedReferenceImage.name ?? "first-frame.jpg",
        }
      : null);

  const stepwisePlaceholder = autoGenerate
    ? displayFirstFrame
      ? t("thread.composer.placeholderStepwiseAuto")
      : t("thread.composer.placeholderStepwiseAutoEmpty")
    : t("thread.composer.placeholderStepwise");

  const handleAnswerQuestion = useCallback(
    (messageId: string, cardId: string, value: string) => {
      const answer = onAnswerQuestion ?? answerQuestion;
      const target = activeMessages.find((item) => item.id === messageId);
      const card = target?.questions?.find((item) => item.id === cardId);
      const intent = classifyFirstFrameQuestion(value, card?.question);
      void (async () => {
        if (intent === "confirm_uploaded" || intent === "confirm_edit_done") {
          const frame = firstFrameValueRef.current;
          if (frame) {
            try {
              await persistStepwiseReference(frame);
            } catch (error) {
              showFirstFrameError(error);
              return;
            }
          } else if (
            intent === "confirm_edit_done" &&
            persistedReferenceImage?.url &&
            session?.key
          ) {
            try {
              await deleteReferenceImage(token, session.key);
              onReferenceImageSynced?.();
            } catch (err) {
              if (err instanceof ApiError && err.status === 409) {
                toast.error(t("thread.firstFrame.locked"));
              }
            }
          }
        }
        answer(messageId, cardId, value);
      })();
    },
    [
      activeMessages,
      answerQuestion,
      onAnswerQuestion,
      onReferenceImageSynced,
      persistStepwiseReference,
      persistedReferenceImage?.url,
      session?.key,
      showFirstFrameError,
      t,
      token,
    ],
  );

  const composerShared = {
    durationSec,
    onDurationChange,
    videoSize,
    onVideoSizeChange,
    videoSizeLocked,
    storyLanguage,
    onStoryLanguageChange,
    autoGenerate,
    onAutoGenerateChange: setAutoGenerate,
    showAutoGenerateToggle: !forceAutoGenerate,
    durationOptions: LONG_VIDEO_DURATION_OPTIONS,
    showFirstFrame: showFirstFrameUploader,
    firstFrame: displayFirstFrame,
    firstFrameCropping,
    firstFrameLocked: referenceImageLocked,
    onFirstFramePick: onPickFirstFrame,
    onFirstFrameClear: onClearFirstFrame,
    onFirstFrameLockedAttempt,
  } as const;

  return (
    <section
      className="relative flex min-h-0 flex-1 flex-col overflow-hidden"
      onDragEnter={showFirstFrameUploader ? onDragEnter : undefined}
      onDragOver={showFirstFrameUploader ? onDragOver : undefined}
      onDragLeave={showFirstFrameUploader ? onDragLeave : undefined}
      onDrop={showFirstFrameUploader ? onDrop : undefined}
    >
      <ThreadHeader
        title={title}
        onToggleSidebar={onToggleSidebar}
        onGoHome={onGoHome}
        hideSidebarToggleOnDesktop={hideSidebarToggleOnDesktop}
        chatId={chatId ?? undefined}
      />
      <ThreadViewport
        messages={activeMessages}
        isStreaming={activeStreaming}
        questionsReady={activeTurnComplete}
        emptyState={emptyState}
        actionSlot={isMockMode ? actionSlot : undefined}
        hideComposer={!!modeSelectorSlot && activeMessages.length === 0}
        onAnswerQuestion={handleAnswerQuestion}
        onMemoryReviewAction={onMemoryReviewAction}
        memoryAssets={memoryAssets}
        onCreateMemoryAsset={onCreateMemoryAsset}
        getNextContinuous={getNextContinuous}
        composer={
          <>
            {!isMockMode && streamError ? (
              <StreamErrorNotice
                error={streamError}
                onDismiss={dismissStreamError}
              />
            ) : null}
            {isMockMode ? (
              <ThreadComposer
                onSend={activeSend}
                disabled={false}
                sendDisabled={composerSendDisabled || !activeTurnComplete}
                placeholder={stepwisePlaceholder}
                modelLabel={toModelBadgeLabel(modelName)}
                variant={showHero ? "hero" : "thread"}
                {...composerShared}
                showFirstFrame={false}
              />
            ) : session ? (
              <ThreadComposer
                onSend={guardedSessionSend}
                disabled={!chatId}
                sendDisabled={composerSendDisabled || !activeTurnComplete}
                placeholder={stepwisePlaceholder}
                modelLabel={toModelBadgeLabel(modelName)}
                variant={showHeroComposer ? "hero" : "thread"}
                {...composerShared}
              />
            ) : (
              <ThreadComposer
                onSend={handleWelcomeSend}
                disabled={booting}
                sendDisabled={composerSendDisabled || !activeTurnComplete}
                placeholder={
                  booting
                    ? t("thread.composer.placeholderOpening")
                    : stepwisePlaceholder
                }
                modelLabel={toModelBadgeLabel(modelName)}
                variant="hero"
                {...composerShared}
              />
            )}
          </>
        }
      />
      {showFirstFrameUploader && isDragging ? (
        <div
          className={cn(
            "pointer-events-none absolute inset-0 z-40 flex items-center justify-center",
            "bg-background/70 backdrop-blur-[2px]",
          )}
          aria-hidden
        >
          <div
            className={cn(
              "rounded-2xl border-2 border-dashed border-foreground/25",
              "bg-background/90 px-8 py-6 text-sm font-medium text-foreground/70",
            )}
          >
            {t("thread.firstFrame.dropHint")}
          </div>
        </div>
      ) : null}
      <AlertDialog
        open={firstFrameUploadFailedOpen}
        onOpenChange={(open) => {
          if (!open) setFirstFrameUploadFailedOpen(false);
        }}
      >
        <AlertDialogContent className="max-w-sm">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[15px]">
              {t("thread.firstFrame.uploadFailed")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {firstFrameUploadError || t("thread.firstFrame.uploadFailed")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="mt-2">
            <AlertDialogAction
              onClick={() => setFirstFrameUploadFailedOpen(false)}
              className="h-8 rounded-lg px-3 text-[13px]"
            >
              {t("thread.firstFrame.uploadFailedOk")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}

function ThreadShellConnected(props: Omit<ThreadShellProps, "stream">) {
  const stream = useThreadStream(props.session, {
    onReplyEnd: props.onReplyEnd,
  });
  return <ThreadShellInner {...props} stream={stream} />;
}

export function ThreadShell({ stream, ...props }: ThreadShellProps) {
  if (stream) {
    return <ThreadShellInner {...props} stream={stream} />;
  }
  return <ThreadShellConnected {...props} />;
}
