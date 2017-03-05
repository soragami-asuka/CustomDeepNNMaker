//=======================================
// バッチ処理データ番号リスト生成クラス
//=======================================
#ifndef __GRAVISBELL_I_BATCH_DATA_NO_LIST_GENERATOR_H__
#define __GRAVISBELL_I_BATCH_DATA_NO_LIST_GENERATOR_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"IOutputLayer.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** 入力データレイヤー */
	class IBatchDataNoListGenerator
	{
	public:
		/** コンストラクタ */
		IBatchDataNoListGenerator(){}
		/** デストラクタ */
		virtual ~IBatchDataNoListGenerator(){}


	public:
		/** 演算前処理を実行する.
			@param dataCount	総データ数
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcess(unsigned int dataCount, unsigned int batchSize) = 0;

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ErrorCode PreProcessLearnLoop() = 0;


	public:
		/** データ数を取得する */
		virtual unsigned int GetDataCount()const = 0;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		virtual unsigned int GetBatchSize()const = 0;


	public:
		/** データ番号リスト数を取得する.
			@return	データ番号リストの総数 = データ数 / バッチサイズ (端数切り上げ)が返る */
		virtual unsigned int GetBatchDataNoListCount()const = 0;

		/** データ番号リストを取得する.
			@param	no	取得するデータ番号リストの番号. 0 <= n < GetBatchDataNoListCount() までの範囲.
			@return	データ番号リストの配列が変える. [GetBatchSize()]の要素数 */
		virtual const unsigned int* GetBatchDataNoListByNum(unsigned int no)const = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif