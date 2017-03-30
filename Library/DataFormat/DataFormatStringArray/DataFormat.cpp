// DataFormatStringArray.cpp : DLL �A�v���P�[�V�����p�ɃG�N�X�|�[�g�����֐����`���܂��B
//

#include "stdafx.h"


#include"DataFormat.h"

#include<string>
#include<vector>
#include<list>
#include<set>
#include<map>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include"Library/Common/StringUtility/StringUtility.h"


namespace Gravisbell {
namespace DataFormat {
namespace StringArray {
	
	struct BoolValue
	{
		F32 trueValue;	/**< true�̎��̒l */
		F32 falseValue;	/**< false�̎��̒l */

		BoolValue()
			:	trueValue	(1.0f)
			,	falseValue	(0.0f)
		{
		}
		BoolValue(F32 trueValue, F32 falseValue)
			:	trueValue	(trueValue)
			,	falseValue	(falseValue)
		{
		}
	};

	/** �f�[�^�t�H�[�}�b�g�̃A�C�e�� */
	class CDataFormatItem
	{
	private:
		const std::wstring id;
		const std::wstring category;

	public:
		/** �R���X�g���N�^ */
		CDataFormatItem(const std::wstring& id, const std::wstring& category)
			:	id	(id)
			,	category	(category)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CDataFormatItem(){}

	public:
		/** ID�̎擾 */
		const std::wstring& GetID()const{return this->id;}
		/** �f�[�^��ʂ̎擾 */
		const std::wstring& GetCategory()const{return this->category;}

		/** �g�p�o�b�t�@����Ԃ� */
		virtual U32 GetBufferCount()const = 0;

		/** �o�b�t�@���擾���� */
		virtual F32 GetBuffer(U32 dataNum, U32 bufferNum)const = 0;

		/** �f�[�^��ǉ����� */
		virtual ErrorCode AddData(const std::wstring& buf) = 0;

		/** ���K�� */
		virtual ErrorCode Normalize() = 0;
	};
	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��.�ŏ��l,�ő�l�w�肠�� */
	class CDataFormatItemMinMaxOutput
	{
	private:
		const F32 minOutput;
		const F32 maxOutput;

	public:
		/** �R���X�g���N�^ */
		CDataFormatItemMinMaxOutput(F32 i_minOutput, F32 i_maxOutput)
			:	minOutput		(i_minOutput)
			,	maxOutput		(i_maxOutput)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CDataFormatItemMinMaxOutput(){}

	public:
		/** �o�͒l�ɕϊ�����
			@param value	0�`1�̒l. */
		F32 CalcOutputValue(F32 value){return min(maxOutput, max(minOutput, value * (maxOutput - minOutput) + minOutput));} 
	};

	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(float�^) */
	class CDataFormatItemFloat : public CDataFormatItem
	{
	protected:
		std::list<F32> lpData;

	public:
		/** �R���X�g���N�^ */
		CDataFormatItemFloat(const std::wstring& id, const std::wstring& category)
			: CDataFormatItem(id, category)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CDataFormatItemFloat()
		{
		}

	public:
		/** �g�p�o�b�t�@����Ԃ� */
		U32 GetBufferCount()const{return 1;}

		/** �o�b�t�@���擾���� */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpData.size())
				return 0.0f;
			if(bufferNum >= 1)
				return 0.0f;

			auto it = this->lpData.begin();
			for(U32 i=0; i<dataNum; i++)
				it++;

			return *it;
		}

		/** �f�[�^��ǉ����� */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back((F32)_wtof(buf.c_str()));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(float�^)(�ŏ��l/�ő�l�w��Ő��K��) */
	class CDataFormatItemFloatNormalizeMinMax : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	public:
		CDataFormatItemFloatNormalizeMinMax(const std::wstring& id, const std::wstring& category, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeMinMax()
		{
		}
		
		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			// �ŏ��l�ƍő�l�����߂�
			F32 minValue = -FLT_MAX;
			F32 maxValue =  FLT_MAX;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					if(*it < minValue)
						minValue = *it;
					if(*it > maxValue)
						maxValue = *it;

					it++;
				}
			}

			// �ŏ��l�ƍő�l��p���Đ��K��
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(float�^)(�w�肳�ꂽ�l�͈̔͂Ő��K������) */
	class CDataFormatItemFloatNormalizeValue : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	private:
		F32 minValue;	/**< �ŏ��l�̒l */
		F32 maxValue;	/**< �ő�l�̒l */

	public:
		CDataFormatItemFloatNormalizeValue(const std::wstring& id, const std::wstring& category, F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	minValue	(i_minValue)
			,	maxValue	(i_maxValue)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeValue()
		{
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			// �ŏ��l�ƍő�l��p���Đ��K��
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(float�^)(�S�f�[�^�̕��ϒl�A�W���΍������ɕW��������) */
	class CDataFormatItemFloatNormalizeAverageDeviation : public CDataFormatItemFloat, public CDataFormatItemMinMaxOutput
	{
	private:
		F32 minValue;	/**< �ŏ��l�̒l */
		F32 maxValue;	/**< �ő�l�̒l */

	public:
		CDataFormatItemFloatNormalizeAverageDeviation(const std::wstring& id, const std::wstring& category, F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemFloat		(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	minValue	(i_minValue)
			,	maxValue	(i_maxValue)
		{
		}
		virtual ~CDataFormatItemFloatNormalizeAverageDeviation()
		{
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			// ���ϒl�����߂�
			F32 average = 0.0f;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					average += *it;
					it++;
				}
				average = average / this->lpData.size();
			}

			// �W���΍�
			F32 deviation = 0.0f;
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					deviation += (*it - average) * (*it - average);
					it++;
				}
				deviation = sqrt(deviation / this->lpData.size());
			}

			// �l���v�Z
			{
				auto it = this->lpData.begin();
				while(it != this->lpData.end())
				{
					F32 value = (*it - minValue) / (maxValue - minValue);

					*it = this->CalcOutputValue(value);

					it++;
				}
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(bool�^) */
	class CDataFormatItemBool : public CDataFormatItem, public CDataFormatItemMinMaxOutput
	{
	protected:
		std::list<std::wstring> lpData;
		std::vector<F32> lpNormalizeValue;	/**< ���K�������f�[�^ */

		std::set<std::wstring> lpFalseString;
		std::set<std::wstring> lpTrueString;

	public:
		CDataFormatItemBool(const std::wstring& id, const std::wstring& category, const std::vector<std::wstring>& i_lpFalseString, const std::vector<std::wstring>& i_lpTrueString, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItem				(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
		{
			for(auto buf : i_lpFalseString)
				this->lpFalseString.insert(buf);
			for(auto buf : i_lpTrueString)
				this->lpTrueString.insert(buf);
		}
		
	public:
		/** �g�p�o�b�t�@����Ԃ� */
		U32 GetBufferCount()const{return 1;}

		/** �o�b�t�@���擾���� */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpNormalizeValue.size())
				return 0.0f;
			if(bufferNum >= 1)
				return 0.0f;

			return lpNormalizeValue[bufferNum];
		}

		/** �f�[�^��ǉ����� */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back(buf.c_str());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			this->lpNormalizeValue.clear();



		}
	};

	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(enum�^) */
	class CDataFormatItemEnumBitArray : public CDataFormatItem, public CDataFormatItemMinMaxOutput
	{
	protected:
		std::list<std::wstring> lpData;
		std::list<std::vector<F32>> lpNormalizeValue;	/**< ���K�������f�[�^ */

		std::vector<std::wstring> lpEnumData;
		std::wstring defaultData;

	public:
		CDataFormatItemEnumBitArray(const std::wstring& id, const std::wstring& category, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItem				(id, category)
			,	CDataFormatItemMinMaxOutput	(i_minOutput, i_maxOutput)
			,	defaultData	(L"")
		{
		}

	public:
		/** �g�p�o�b�t�@����Ԃ� */
		U32 GetBufferCount()const{return this->lpEnumData.size();}

		/** �o�b�t�@���擾���� */
		F32 GetBuffer(U32 dataNum, U32 bufferNum)const
		{
			if(dataNum >= this->lpNormalizeValue.size())
				return 0.0f;

			// �f�[�^�ʒu���ړ�
			auto it_data = this->lpNormalizeValue.begin();
			for(U32 i=0; i<dataNum; i++)
				it_data++;

			// �o�b�t�@�����m�F
			if(bufferNum >= it_data->size())
				return 0.0f;

			// �o�b�t�@�ʒu���ړ�
			auto it_buf = it_data->begin();
			for(U32 i=0; i<bufferNum; i++)
				it_buf++;

			return *it_buf;
		}

		/** �f�[�^��ǉ����� */
		ErrorCode AddData(const std::wstring& buf)
		{
			this->lpData.push_back(buf.c_str());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			// �g�p����Ă��镶�����񋓂���
			std::set<std::wstring> lpTmpEnumData;
			for(auto& data : this->lpData)
			{
				lpTmpEnumData.insert(data);
			}

			this->lpEnumData.clear();
			for(auto& data: lpTmpEnumData)
			{
				this->lpEnumData.push_back(data);
			}

			return this->NormalizeData();
		}
		/** ���K�� */
		ErrorCode NormalizeData()
		{
			// �f�t�H���g�������l�ɕϊ�
			U32 defaultValue = 0;
			{
				U32 value=0;
				for(auto enumValue : lpEnumData)
				{
					if(enumValue == defaultData)
					{
						defaultValue = value;
						break;
					}
					value++;
				}
			}

			// �l�𐳋K������
			this->lpNormalizeValue.clear();
			for(auto& data : this->lpData)
			{
				// ���K����̒l��ۑ����邽�߂̃o�b�t�@���m��
				std::vector<F32> lpValue(this->lpEnumData.size());
				if(defaultValue < lpValue.size())
					lpValue[defaultValue] = this->CalcOutputValue(1.0f);

				// enum�l�̒l���擾����
				U32 value=0;
				for(auto enumValue : this->lpEnumData)
				{
					if(enumValue == data)
					{
						lpValue[value] = this->CalcOutputValue(1.0f);
					}
					else
					{
						lpValue[value] = this->CalcOutputValue(0.0f);
					}
					value++;
				}

				// �o�b�t�@��}��
				this->lpNormalizeValue.push_back(lpValue);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};
	/** �f�[�^�t�H�[�}�b�g�̃A�C�e��(enum�^) */
	class CDataFormatItemEnumBitArrayEnum : public CDataFormatItemEnumBitArray
	{
	public:
		CDataFormatItemEnumBitArrayEnum(const std::wstring& id, const std::wstring& category, const std::vector<std::wstring>& i_lpEnumString, std::wstring& i_defaultData, F32 i_minOutput, F32 i_maxOutput)
			:	CDataFormatItemEnumBitArray(id, category, i_minOutput, i_maxOutput)
		{
			this->defaultData = i_defaultData;
			this->lpEnumData = i_lpEnumString;
		}

		/** ���K�� */
		virtual ErrorCode Normalize()
		{
			return this->NormalizeData();
		}
	};

	/** �f�[�^�t�H�[�}�b�g */
	class CDataFormat : public IDataFormat
	{
	private:
		std::wstring name;	/**< ���O */
		std::wstring text;	/**< ������ */

		std::set<std::wstring> lpCategoryName;	/**< �f�[�^��ʖ��ꗗ */

		U32 dataCount;	/**< �f�[�^�� */

		std::map<std::wstring, std::vector<F32>> lpTmpOutput;	/**< �o�̓f�[�^�i�[�p�̈ꎞ�o�b�t�@ */
		std::vector<CDataFormatItem*> lpDataFormat;	/**< �f�[�^�t�H�[�}�b�g�̈ꗗ */

	public:
		/** �R���X�g���N�^ */
		CDataFormat()
		:	CDataFormat(L"", L"")
		{
		}
		/** �R���X�g���N�^ */
		CDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
		:	name		(i_szName)
		,	text		(i_szText)
		,	dataCount	(0)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~CDataFormat()
		{
			// �f�[�^�t�H�[�}�b�g��S�폜
			this->ClearDataFormat();
		}

	public:
		/** ���O�̎擾 */
		const wchar_t* GetName()const
		{
			return name.c_str();
		}
		/** �������̎擾 */
		const wchar_t* GetText()const
		{
			return text.c_str();
		}

		/** X�����̗v�f�����擾 */
		U32 GetBufferCountX(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Y�����̗v�f�����擾 */
		U32 GetBufferCountY(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** Z�����̗v�f�����擾 */
		U32 GetBufferCountZ(const wchar_t i_szCategory[])const
		{
			return 1;
		}
		/** CH�����̗v�f�����擾 */
		U32 GetBufferCountCH(const wchar_t i_szCategory[])const
		{
			U32 bufferCount = 0;
			for(auto it : this->lpDataFormat)
			{
				if(it == NULL)
					continue;

				// �J�e�S�����`�F�b�N
				if(it->GetCategory() != i_szCategory)
					continue;

				bufferCount += it->GetBufferCount();
			}

			return bufferCount;
		}

		/** �f�[�^�\�����擾 */
		IODataStruct GetDataStruct(const wchar_t i_szCategory[])const
		{
			return IODataStruct(this->GetBufferCountCH(i_szCategory), this->GetBufferCountX(i_szCategory), this->GetBufferCountY(i_szCategory), this->GetBufferCountZ(i_szCategory));
		}

		/** �J�e�S���[�����擾���� */
		U32 GetCategoryCount()const
		{
			return this->lpCategoryName.size();
		}
		/** �J�e�S���[����ԍ��w��Ŏ擾���� */
		const wchar_t* GetCategoryNameByNum(U32 categoryNo)const
		{
			if(categoryNo >= this->lpCategoryName.size())
				return NULL;

			auto it = this->lpCategoryName.begin();
			for(U32 no=0; no<categoryNo; no++)
				it++;
			return it->c_str();
		}

	public:
		/** �f�[�^�����擾���� */
		U32 GetDataCount()const
		{
			return this->dataCount;
		}

		/** �f�[�^���擾���� */
		const F32* GetDataByNum(U32 i_dataNo, const wchar_t i_szCategory[])const
		{
			if(i_dataNo >= this->dataCount)
				return NULL;

			auto it = this->lpTmpOutput.find(i_szCategory);
			if(it == this->lpTmpOutput.end())
				return NULL;

			return &it->second[0];
		}

	public:
		/** ���K������.
			�f�[�^�̒ǉ����I��������A��x�̂ݎ��s. ��������s����ƒl�����������Ȃ�̂Œ���. */
		Gravisbell::ErrorCode Normalize()
		{
			for(auto it : this->lpDataFormat)
				it->Normalize();

			// �o�͗p�ꎞ�o�b�t�@���m��
			this->lpTmpOutput.clear();
			for(auto& categoryName : this->lpCategoryName)
			{
				this->lpTmpOutput[categoryName].resize(this->GetBufferCountCH(categoryName.c_str()));
			}

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** �f�[�^�t�H�[�}�b�g�����擾���� */
		U32 GetDataFormatCount()const
		{
			return this->lpDataFormat.size();
		}

		/** �f�[�^�t�H�[�}�b�g��S�폜���� */
		Gravisbell::ErrorCode ClearDataFormat()
		{
			for(auto it : this->lpDataFormat)
			{
				delete it;
			}
			this->lpDataFormat.clear();

			return ErrorCode::ERROR_CODE_NONE;
		}

		//=============================================
		// float�^
		//=============================================
		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K���Ȃ�
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���. */
		Gravisbell::ErrorCode AddDataFormatFloat(const wchar_t i_szID[], const wchar_t i_szCategory[])
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloat(i_szID, i_szCategory));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����.
			�S�f�[�^�̍ŏ��l�A�ő�l�Ő��K��
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���.
			@param	i_minOutput		�o�͂����ŏ��l.
			@param	i_maxOutput		�o�͂����ő�l. */
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeMinMax(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeMinMax(i_szID, i_szCategory, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����.
			i_minValue, i_maxValue �Ő��K��. �o�͂����l��i_minOutput, i_maxOutput�̊ԂɂȂ�.
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���.
			@param	i_minValue		�f�[�^���̍ŏ��l.
			@param	i_maxValue		�f�[�^���̍ő�l.
			@param	i_minOutput		�o�͂����ŏ��l.
			@param	i_maxOutput		�o�͂����ő�l. */
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeValue(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeValue(i_szID, i_szCategory, i_minValue, i_maxValue, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����.
			���ϒl�ƕW���΍������ɕW��������.
			���Z����-���U �� [i_minValue]
			���Z����+���U �� [i_maxValue]
			�ɂȂ�悤�������A
			i_minValue -> i_minOutput
			i_maxValue -> i_maxOutput
			�ɂȂ�悤�ɐ��K������
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���.
			@param	i_minValue		�v�Z���ʂ̍ŏ��l.
			@param	i_maxValue		�v�Z���ʂ̍ő�l.
			@param	i_minOutput		�o�͂����ŏ��l.
			@param	i_maxOutput		�o�͂����ő�l. */
		Gravisbell::ErrorCode AddDataFormatFloatNormalizeAverageDeviation(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemFloatNormalizeAverageDeviation(i_szID, i_szCategory, i_minValue, i_maxValue, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}


		//=============================================
		// string�^
		//=============================================
		/** string�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K������1,0�̔z��ɕϊ�����
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���. */
		Gravisbell::ErrorCode AddDataFormatStringToBitArray(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput)
		{
			this->lpDataFormat.push_back(new CDataFormatItemEnumBitArray(i_szID, i_szCategory, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** string�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K������Enum�l�����ɂ���1,0�̔z��ɕϊ�����.
			@param	i_szID				����ID.
			@param	i_szCategory		�f�[�^���.
			@param	i_enumValueCount	enum�l�̐�.
			@param	i_lpEnumString		enum�l�̕�����̔z��.
			@param	i_defaultValue		���̓f�[�^�ɏ���̒l�������Ă��Ȃ������ꍇ�ɐݒ肳���f�t�H���g�l. */
		Gravisbell::ErrorCode AddDataFormatStringToBitArrayEnum(const wchar_t i_szID[], const wchar_t i_szCategory[], U32 i_enumDataCount, const wchar_t*const i_lpEnumData[], const wchar_t i_defaultData[], F32 i_minOutput, F32 i_maxOutput)
		{
			std::vector<std::wstring> lpEnumData;
			for(U32 i=0; i<i_enumDataCount; i++)
				lpEnumData.push_back(i_lpEnumData[i]);

			this->lpDataFormat.push_back(new CDataFormatItemEnumBitArrayEnum(i_szID, i_szCategory, lpEnumData, (std::wstring)i_defaultData, i_minOutput, i_maxOutput));

			return ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** �f�[�^�𕶎���z��Œǉ����� */
		Gravisbell::ErrorCode AddDataByStringArray(const wchar_t*const i_szBuffer[])
		{
			for(U32 i=0; i<this->lpDataFormat.size(); i++)
			{
				this->lpDataFormat[i]->AddData(i_szBuffer[i]);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormat(const wchar_t i_szName[], const wchar_t i_szText[])
	{
		return new CDataFormat(i_szName, i_szText);
	}
	/** ������̔z���ǂݍ��ރf�[�^�t�H�[�}�b�g���쐬���� */
	extern GRAVISBELL_LIBRAY_API IDataFormat* CreateDataFormatFromXML(const wchar_t szXMLFilePath[])
	{
		using namespace StringUtility;

		// XML�t�@�C���̓ǂݍ���
		boost::property_tree::ptree pXmlTree;
		boost::iostreams::file_descriptor_source fs(UnicodeToShiftjis(szXMLFilePath));
		boost::iostreams::stream<boost::iostreams::file_descriptor_source> fsstream(fs);
		try
		{
			boost::property_tree::read_xml(fsstream, pXmlTree);
		}
		catch(boost::exception& e)
		{
			e;
			return NULL;
		}

		CDataFormat* pDataFormat = NULL;
		try
		{
			// ���O
			std::wstring name;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Name"))
			{
				name = UTF8toUnicode(pValue.get());
			}
			// �e�L�X�g
			std::wstring text;
			if(boost::optional<std::string> pValue = pXmlTree.get_optional<std::string>("DataFormat.Text"))
			{
				text = UTF8toUnicode(pValue.get());
			}

			// bool�l�̒l
			std::map<std::wstring, BoolValue>	lpBoolValue;	/**< bool�l��F32�ɕϊ�����ݒ�l�̈ꗗ.	<�f�[�^��ʖ�, �ϊ��f�[�^> */
			lpBoolValue[L""] = BoolValue();
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.BoolValue"))
			{
				if(it.first == "true" || it.first == "false")
				{
					// ��������J�e�S�����擾
					std::wstring category = L"";
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
					{
						category = UTF8toUnicode(pValue.get());
					}

					if(it.first == "true")
						lpBoolValue[category].trueValue = (F32)atof(it.second.data().c_str());
					else if(it.first == "false")
						lpBoolValue[category].falseValue = (F32)atof(it.second.data().c_str());
				}
			}


			// �f�[�^�t�H�[�}�b�g���쐬
			pDataFormat = new CDataFormat();


			// Channel�̓ǂݍ���
			for(const boost::property_tree::ptree::value_type &it : pXmlTree.get_child("DataFormat.Channel"))
			{
				// id�̎擾
				std::wstring id = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.id"))
				{
					id = UTF8toUnicode(pValue.get());
				}
				// category�̎擾
				std::wstring category = L"";
				if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.category"))
				{
					category = UTF8toUnicode(pValue.get());
				}

				// bool�^�̒l���擾
				BoolValue boolValue = lpBoolValue[L""];
				if(lpBoolValue.count(category))
					boolValue = lpBoolValue[category];

				if(it.first == "String")
				{
					enum UseType
					{
						USETYPE_BITARRAY,
						USETYPE_BITARRAY_ENUM
					};
					UseType useType = USETYPE_BITARRAY;

					// �g�p���@���擾
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.useType"))
					{
						if(pValue.get() == "bit_array")
							useType = USETYPE_BITARRAY;
						else if(pValue.get() == "bit_array_enum")
							useType = USETYPE_BITARRAY_ENUM;
					}

					switch(useType)
					{
					case USETYPE_BITARRAY:
					default:
						{
							// �t�H�[�}�b�g��ǉ�
							pDataFormat->AddDataFormatStringToBitArray(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;
					case USETYPE_BITARRAY_ENUM:
						{
							// enum�l���
							std::list<std::wstring> lpEnumString;
							std::vector<const wchar_t*> lpEnumStringPointer;
							std::wstring defaultString = L"";
							if(auto& pTreeEnum = it.second.get_child_optional("Enum"))
							{
								for(const boost::property_tree::ptree::value_type &it_enum : pTreeEnum.get().get_child(""))
								{
									if(it_enum.first == "item")
									{
										lpEnumString.push_back(UTF8toUnicode(it_enum.second.data()));
										lpEnumStringPointer.push_back(lpEnumString.rbegin()->c_str());

										if(boost::optional<std::string> pValue = it_enum.second.get_optional<std::string>("<xmlattr>.default"))
										{
											if(pValue.get() == "true")
												defaultString = UTF8toUnicode(it_enum.second.data());
										}
									}
								}
							}

							// �t�H�[�}�b�g��ǉ�
							pDataFormat->AddDataFormatStringToBitArrayEnum(id.c_str(), category.c_str(), lpEnumStringPointer.size(), &lpEnumStringPointer[0], defaultString.c_str(), boolValue.falseValue, boolValue.trueValue); 
						}
						break;
					}
				}
				else if(it.first == "Float")
				{
					enum NormalizeType
					{
						NORMALIZETYPE_NONE,		// ���K�����Ȃ�
						NORMALIZETYPE_MINMAX,	// �S�f�[�^�̍ŏ��l�A�ő�l�����ɐ��K������
						NORMALIZETYPE_VALUE,	// �ŏ��l�A�ő�l���w�肵�Đ��K������
						NORMALIZETYPE_SDEV,		// �S�f�[�^�̕��ϒl�A�W���΍������ɐ��K������
					};
					NormalizeType normalizeType = NORMALIZETYPE_NONE;

					// ���K����ʂ��擾
					if(boost::optional<std::string> pValue = it.second.get_optional<std::string>("<xmlattr>.normalize"))
					{
						if(pValue.get() == "none")
							normalizeType = NORMALIZETYPE_NONE;
						else if(pValue.get() == "min_max")
							normalizeType = NORMALIZETYPE_MINMAX;
						else if(pValue.get() == "value")
							normalizeType = NORMALIZETYPE_VALUE;
						else if(pValue.get() == "average_deviation")
							normalizeType = NORMALIZETYPE_SDEV;
					}

					// �ݒ�ŏ��l, �ő�l���擾����
					F32 minValue = 0.0f;
					F32 maxValue = 1.0f;
					if(boost::optional<float> pValue = it.second.get_optional<float>("min"))
						minValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("max"))
						maxValue = pValue.get();

					// �o�͍ŏ��l�A�ő�l���擾����
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_min"))
						boolValue.falseValue = pValue.get();
					if(boost::optional<float> pValue = it.second.get_optional<float>("output_max"))
						boolValue.trueValue = pValue.get();

					switch(normalizeType)
					{
					case NORMALIZETYPE_NONE:		// ���K�����Ȃ�
					default:
						pDataFormat->AddDataFormatFloat(id.c_str(), category.c_str());
						break;
					case NORMALIZETYPE_MINMAX:	// �S�f�[�^�̍ŏ��l�A�ő�l�����ɐ��K������
						pDataFormat->AddDataFormatFloatNormalizeMinMax(id.c_str(), category.c_str(), boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_VALUE:	// �ŏ��l�A�ő�l���w�肵�Đ��K������
						pDataFormat->AddDataFormatFloatNormalizeValue(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					case NORMALIZETYPE_SDEV:		// �S�f�[�^�̕��ϒl�A�W���΍������ɐ��K������
						pDataFormat->AddDataFormatFloatNormalizeAverageDeviation(id.c_str(), category.c_str(), minValue, maxValue, boolValue.falseValue, boolValue.trueValue);
						break;
					}
				}
			}
		}
		catch(boost::exception& e)
		{
			e;
			if(pDataFormat)
				delete pDataFormat;
			return NULL;
		}

		return pDataFormat;
	}


}	// StringArray
}	// DataFormat
}	// Gravisbell


