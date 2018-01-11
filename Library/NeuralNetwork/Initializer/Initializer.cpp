// Initializer.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
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
		InitializerID_zero,		// �S��0
		InitializerID_one,		// �S��1

		InitializerID_uniform,	// -1�`+1�̈�l����
		InitializerID_normal,	// average=0.0, variance=1.0�̐��K����

		InitializerID_glorot_normal,	// Glorot�̐��K���z. stddev = sqrt(2 / (fan_in + fan_out))�̐ؒf���K���z
		InitializerID_glorot_uniform,	// Glorot�̈�l���z. limit  = sqrt(6 / (fan_in + fan_out))�̈�l���z
		InitializerID_he_normal,		// He�̐��K���z. stddev = sqrt(2 / fan_in)�̐ؒf���K���z
		InitializerID_he_uniform,		// He�̈�l���z. limit  = sqrt(6 / fan_in)�̈�l���z
		InitializerID_lecun_uniform,	// LeCun�̈�l���z. limit = sqrt(3 / fan_in)�̈�l���z
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
		/** �R���X�g���N�^ */
		InitializerManager()
			:	IInitializerManager()
			,	random()
			,	pCurrentInitializer	(NULL)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~InitializerManager()
		{
			SAFE_DELETE(this->pCurrentInitializer);
		}

	public:
		/** ���������������� */
		ErrorCode InitializeRandomParameter()
		{
			this->random.Initialize((U32)time(NULL));

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** ���������������� */
		ErrorCode InitializeRandomParameter(U32 i_seed)
		{
			this->random.Initialize(i_seed);

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �������N���X���擾���� */
		IInitializer& GetInitializer(const wchar_t i_initializerID[])
		{
			// �������ɕϊ�
			std::wstring initializerID_buf = i_initializerID;
			std::transform(initializerID_buf.begin(), initializerID_buf.end(), initializerID_buf.begin(), towlower);

			// ID�ɕϊ�
			InitializerID initializerID = InitializerID_glorot_uniform;
			auto it_id = lpInitializerCode2ID.find(initializerID_buf);
			if(it_id != lpInitializerCode2ID.end())
				initializerID = it_id->second;

			// �N���X���擾
			switch(initializerID)
			{
			case InitializerID_zero:	// �S��0
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_zero();
				break;

			case InitializerID_one:		// �S��1
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_one();
				break;

			case InitializerID_uniform:	// -1�`+1�̈�l����
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_uniform(this->random);
				break;

			case InitializerID_normal:	// average=0.0, variance=1.0�̐��K����
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_normal(this->random);
				break;

			case InitializerID_glorot_normal:	// Glorot�̐��K���z. stddev = sqrt(2 / (fan_in + fan_out))�̐ؒf���K���z
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_glorot_normal(this->random);
				break;

			case InitializerID_glorot_uniform:	// Glorot�̈�l���z. limit  = sqrt(6 / (fan_in + fan_out))�̈�l���z
			default:
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_glorot_uniform(this->random);
				break;

			case InitializerID_he_normal:		// He�̐��K���z. stddev = sqrt(2 / fan_in)�̐ؒf���K���z
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_he_normal(this->random);
				break;

			case InitializerID_he_uniform:		// He�̈�l���z. limit  = sqrt(6 / fan_in)�̈�l���z
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_he_uniform(this->random);
				break;

			case InitializerID_lecun_uniform:	// LeCun�̈�l���z. limit = sqrt(3 / fan_in)�̈�l���z
				SAFE_DELETE(this->pCurrentInitializer);
				this->pCurrentInitializer = new Initializer_lecun_uniform(this->random);
				break;
			}

			return *this->pCurrentInitializer;
		}
	};

	/** �������Ǘ��N���X���擾���� */
	Initializer_API IInitializerManager& GetInitializerManager(void)
	{
		static InitializerManager initializerManager;

		return initializerManager;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
