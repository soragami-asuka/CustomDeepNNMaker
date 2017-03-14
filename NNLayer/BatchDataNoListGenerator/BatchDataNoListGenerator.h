#ifdef BATCHDATANOLISTGENERATOR_EXPORTS
#define BatchDataNoListGenerator_API __declspec(dllexport)
#else
#define BatchDataNoListGenerator_API __declspec(dllimport)
#endif

#include"Common/IBatchDataNoListGenerator.h"


/** バッチ処理データ番号リスト生成クラスを作成する. CPU制御 */
extern "C" BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGeneratorCPU();

/** バッチ処理データ番号リスト生成クラスを作成する. GPU制御 */
extern "C" BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGeneratorGPU();
