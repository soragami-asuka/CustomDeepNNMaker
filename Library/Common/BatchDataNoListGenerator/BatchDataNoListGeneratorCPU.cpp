// BatchDataNoListGenerator.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "BatchDataNoListGenerator.h"

#include<vector>
#include<algorithm>
#include <random>

using namespace Gravisbell;

namespace Gravisbell {
namespace Common {

	class BatchDataNoListGenerator : public Gravisbell::Common::IBatchDataNoListGenerator
	{
	private:
		unsigned int dataCount;
		unsigned int batchSize;

		std::vector<unsigned int> lpAllDataNoList;		// 全データ番号のランダム配列
		std::vector<unsigned int>::iterator it_addDataBegin;		// 端数分の開始イテレータ

		std::random_device seed_gen;
		std::mt19937 random_generator;

	public:
		/** コンストラクタ */
		BatchDataNoListGenerator()
			:	IBatchDataNoListGenerator	()
			,	seed_gen					()
			,	random_generator			(seed_gen())
		{
		}
		/** デストラクタ */
		virtual ~BatchDataNoListGenerator()
		{
		}


	public:
		/** 演算前処理を実行する.
			@param dataCount	総データ数
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcess(unsigned int dataCount, unsigned int batchSize)
		{
			this->dataCount = dataCount;
			this->batchSize = batchSize;

			// 配列初期化
			this->lpAllDataNoList.resize( (dataCount + (batchSize-1)) / batchSize * batchSize );
			for(unsigned int i=0; i<dataCount; i++)
			{
				this->lpAllDataNoList[i] = i;
			}

			// 端数の開始イテレータを取得
			this->it_addDataBegin = this->lpAllDataNoList.begin();
			for(unsigned int i=0; i<dataCount; i++)
				this->it_addDataBegin++;

			// 端数部分を穴埋め
			U32 addDataCount = (U32)this->lpAllDataNoList.size() - this->dataCount;
			for(unsigned int i=0; i<addDataCount; i++)
			{
				this->lpAllDataNoList[this->dataCount + i] = this->lpAllDataNoList[i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearnLoop()
		{
			// シャッフル
			std::shuffle(this->lpAllDataNoList.begin(), this->it_addDataBegin, random_generator);

			//// 端数部分を穴埋め
			//U32 addDataCount = (U32)this->lpAllDataNoList.size() - this->dataCount;
			//for(unsigned int i=0; i<addDataCount; i++)
			//{
			//	this->lpAllDataNoList[this->dataCount + i] = this->lpAllDataNoList[i];
			//}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** データ数を取得する */
		unsigned int GetDataCount()const
		{
			return this->dataCount;
		}

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		unsigned int GetBatchSize()const
		{
			return this->batchSize;
		}


	public:
		/** データ番号リスト数を取得する.
			@return	データ番号リストの総数 = データ数 / バッチサイズ (端数切り上げ)が返る */
		unsigned int GetBatchDataNoListCount()const
		{
			return (U32)this->lpAllDataNoList.size() / this->GetBatchSize();
		}

		/** データ番号リストを取得する.
			@param	no	取得するデータ番号リストの番号. 0 <= n < GetBatchDataNoListCount() までの範囲.
			@return	データ番号リストの配列が変える. [GetBatchSize()]の要素数 */
		const unsigned int* GetBatchDataNoListByNum(unsigned int no)const
		{
			return &this->lpAllDataNoList[no * this->batchSize];
		}
	};


	/** バッチ処理データ番号リスト生成クラスを作成する. */
	extern BatchDataNoListGenerator_API Gravisbell::Common::IBatchDataNoListGenerator* CreateBatchDataNoListGenerator()
	{
		return new BatchDataNoListGenerator();
	}

}	// Common
}	// Gravisbell
