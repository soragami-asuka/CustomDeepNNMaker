#ifdef BATCHDATANOLISTGENERATOR_EXPORTS
#define BatchDataNoListGenerator_API __declspec(dllexport)
#else
#define BatchDataNoListGenerator_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Common.BatchDataNoListGenerator.lib")
#endif
#endif

#include"../../Common/IBatchDataNoListGenerator.h"



namespace Gravisbell {
namespace Common {

	/** バッチ処理データ番号リスト生成クラスを作成する. CPU制御 */
	extern BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGenerator();

}	// Common
}	// Gravisbell