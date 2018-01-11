// Initializer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"

#include<map>
#include<string>
#include<algorithm>
#include<time.h>

#include"Library/NeuralNetwork/Initializer.h"
#include"RandomUtility.h"

#include"Initializer_zero.h"
#include"Initializer_one.h"

#include"Initializer_uniform.h"
#include"Initializer_normal.h"

#include"Initializer_glorot_normal.h"
#include"Initializer_glorot_uniform.h"
#include"Initializer_he_normal.h"
#include"Initializer_he_uniform.h"
#include"Initializer_lecun_uniform.h"



#define SAFE_DELETE(p)	{if(p){delete p;}p=NULL;}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	enum InitializerID
	{
		InitializerID_zero,		// 全て0
		InitializerID_one,		// 全て1

		InitializerID_uniform,	// -1～+1の一様乱数
		InitializerID_normal,	// average=0.0, variance=1.0の正規乱数

		InitializerID_glorot_normal,	// Glorotの正規分布. stddev = sqrt(2 / (fan_in + fan_out))の切断正規分布
		InitializerID_glorot_uniform,	// Glorotの一様分布. limit  = sqrt(6 / (fan_in + fan_out))の一様分布
		InitializerID_he_normal,		// Heの正規分布. stddev = sqrt(2 / fan_in)の切断正規分布
		InitializerID_he_uniform,		// Heの一様分布. limit  = sqrt(6 / fan_in)の一様分布
		InitializerID_lecun_uniform,	// LeCunの一様分布. limit = sqrt(3 / fan_in)の一様分布
	};

	static const std::map<std::wstring, InitializerID> lpInitializerCode2ID = 
	{
		{L"zero",				InitializerID_zero	},
		{L"one",				InitializerID_one	},

		{L"uniform",			InitializerID_uniform	},
		{L"normal",				InitializerID_normal	},

		{L"glorot_normal",		InitializerID_glorot_normal	},
		{L"glorot_uniform",		InitializerID_glorot_uniform	},
		{L"he_normal",			InitializerID_he_normal	},
		{L"he_uniform",			InitializerID_he_uniform	},
		{L"lecun_uniform",		InitializerID_lecun_uniform	},
	};

	class InitializerManager : public IInitializerManager
	{
	private:
		Random random;
		IInitializer* pCurrentInitializer;

	public:
		/** コンストラクタ */
		InitializerManager()
			:	IInitializerManager()
			,	random()
			,	pCurrentInitializer	(NULL)
		{
		}
		/** デストラクタ */
		virtual ~InitializerManager()
		{
			SAFE_DELETE(this->pCurrentInitializer);
		}

	public:
		/** 乱数を初期化する */
		ErrorCode InitializeRandomParameter()
		{
			this->random.Initialize((U32)time(NULL));

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** 乱数を初期化する */
		ErrorCode InitializeRandomParameter(U32 i_seed)
		{
			this->random.Initialize(i_seed);

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 初期化クラスを取得する */
		IInitializer& GetInitializer(const wchar_t i_initializerID[])
		{
			// 小文字に変換
			std::wstring initializerID_buf = i_initializerID;
			std::transform(initializerID_buf.begin(), initializerID_buf.end(), initializerID_buf.begin(), towlower);

			// IDに変換
			InitializerID initializerID = InitializerID_glorot_uniform;
			auto it_id = lpInitializerCode2ID.find(initializerID_buf);
			if(it_id != lpInitializerCode2ID.end())
				initializerID = it_id->second;

			// クラスを取得
			switch(initializerID)
			{
			case InitializerID_zero:	// 全て0
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_zero();
				break;

			case InitializerID_one:		// 全て1
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_one();
				break;

			case InitializerID_uniform:	// -1～+1の一様乱数
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_uniform(this->random);
				break;

			case InitializerID_normal:	// average=0.0, variance=1.0の正規乱数
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_normal(this->random);
				break;

			case InitializerID_glorot_normal:	// Glorotの正規分布. stddev = sqrt(2 / (fan_in + fan_out))の切断正規分布
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_glorot_normal(this->random);
				break;

			case InitializerID_glorot_uniform:	// Glorotの一様分布. limit  = sqrt(6 / (fan_in + fan_out))の一様分布
			default:
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_glorot_uniform(this->random);
				break;

			case InitializerID_he_normal:		// Heの正規分布. stddev = sqrt(2 / fan_in)の切断正規分布
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_he_normal(this->random);
				break;

			case InitializerID_he_uniform:		// Heの一様分布. limit  = sqrt(6 / fan_in)の一様分布
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_he_uniform(this->random);
				break;

			case InitializerID_lecun_uniform:	// LeCunの一様分布. limit = sqrt(3 / fan_in)の一様分布
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_lecun_uniform(this->random);
				break;
			}

			return *this->pCurrentInitializer;
		}
	};

	/** 初期化管理クラスを取得する */
	Initializer_API IInitializerManager& GetInitializerManager(void)
	{
		static InitializerManager initializerManager;

		return initializerManager;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
