//author: 代码医生工作室
//公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
//来源: <深度学习之TensorFlow工程化项目实战>配套代码 （700+页）
//配套代码技术支持：bbs.aianaconda.com      (有问必答)

#import "13-3 CameraExampleAppDelegate.h"
@implementation _3_3_CameraExampleAppDelegate

@synthesize window = _window;

- (BOOL)application:(UIApplication *)application
    didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
  [self.window makeKeyAndVisible];
  return YES;
}

- (void)applicationWillResignActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:NO];
}

- (void)applicationDidEnterBackground:(UIApplication *)application {
}

- (void)applicationWillEnterForeground:(UIApplication *)application {
}

- (void)applicationDidBecomeActive:(UIApplication *)application {
  [[UIApplication sharedApplication] setIdleTimerDisabled:YES];
}

- (void)applicationWillTerminate:(UIApplication *)application {
}

@end
