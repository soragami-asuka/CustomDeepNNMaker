//=======================================
// �f�[�^�t�H�[�}�b�g�̕�����z��`���C���^�[�t�F�[�X
//=======================================
#ifndef __GRAVISBELL_I_DATAFORMAT_STRINGARRAY_H__
#define __GRAVISBELL_I_DATAFORMAT_STRINGARRAY_H__

#include"../IDataFormatBase.h"

namespace Gravisbell {
namespace DataFormat {
namespace StringArray {

	/** ���K�����@ */
	enum ENormalizeType
	{
		NORMALIZE_TYPE_NONE,				/**< ���K���Ȃ� */
		NORMALIZE_TYPE_MINMAX,				/**< �S�f�[�^�̍ŏ��l,�ő�l�Ő��K��. */
		NORMALIZE_TYPE_VALUE,				/**< �w�肳�ꂽ�l�Ő��K������. */
		NORMALIZE_TYPE_AVERAGE_DEVIATION,	/**< �S�f�[�^�̕��ϒl, �W���΍������ɐ��K�� */
	};


	/** �f�[�^�t�H�[�}�b�g */
	class IDataFormat : public IDataFormatBase
	{
	public:
		/** �R���X�g���N�^ */
		IDataFormat(){}
		/** �f�X�g���N�^ */
		virtual ~IDataFormat(){}

	public:
		/** �f�[�^�t�H�[�}�b�g�����擾���� */
		virtual U32 GetDataFormatCount()const = 0;

		/** �f�[�^�t�H�[�}�b�g��S�폜���� */
		virtual Gravisbell::ErrorCode ClearDataFormat() = 0;

		//=============================================
		// float�^
		//=============================================
		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K���Ȃ�
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���. */
		virtual Gravisbell::ErrorCode AddDataFormatFloat(const wchar_t i_szID[], const wchar_t i_szCategory[]) = 0;

		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����.
			�S�f�[�^�̍ŏ��l�A�ő�l�Ő��K��
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���.
			@param	i_minOutput		�o�͂����ŏ��l.
			@param	i_maxOutput		�o�͂����ő�l. */
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeMinMax(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput) = 0;

		/** Float�^�f�[�^�t�H�[�}�b�g��ǉ�����.
			i_minValue, i_maxValue �Ő��K��. �o�͂����l��i_minOutput, i_maxOutput�̊ԂɂȂ�.
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���.
			@param	i_minValue		�f�[�^���̍ŏ��l.
			@param	i_maxValue		�f�[�^���̍ő�l.
			@param	i_minOutput		�o�͂����ŏ��l.
			@param	i_maxOutput		�o�͂����ő�l. */
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeValue(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput) = 0;

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
		virtual Gravisbell::ErrorCode AddDataFormatFloatNormalizeAverageDeviation(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minValue, F32 i_maxValue, F32 i_minOutput, F32 i_maxOutput) = 0;



		//=============================================
		// string�^
		//=============================================
		/** string�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K������1,0�̔z��ɕϊ�����
			@param	i_szID			����ID.
			@param	i_szCategory	�f�[�^���. */
		virtual Gravisbell::ErrorCode AddDataFormatStringToBitArray(const wchar_t i_szID[], const wchar_t i_szCategory[], F32 i_minOutput, F32 i_maxOutput) = 0;
		/** string�^�f�[�^�t�H�[�}�b�g��ǉ�����. ���K������Enum�l�����ɂ���1,0�̔z��ɕϊ�����.
			@param	i_szID				����ID.
			@param	i_szCategory		�f�[�^���.
			@param	i_enumValueCount	enum�l�̐�.
			@param	i_lpEnumString		enum�l�̕�����̔z��.
			@param	i_defaultValue		���̓f�[�^�ɏ���̒l�������Ă��Ȃ������ꍇ�ɐݒ肳���f�t�H���g�l. */
		virtual Gravisbell::ErrorCode AddDataFormatStringToBitArrayEnum(const wchar_t i_szID[], const wchar_t i_szCategory[], U32 i_enumDataCount, const wchar_t*const i_lpEnumData[], const wchar_t i_defaultData[], F32 i_minOutput, F32 i_maxOutput) = 0;


	public:
		/** �f�[�^�𕶎���z��Œǉ����� */
		virtual Gravisbell::ErrorCode AddDataByStringArray(const wchar_t*const i_szBuffer[]) = 0;
	};

}	// StringArray
}	// DataFormat
}	// Gravisbell



#endif // __GRAVISBELL_I_DATAFORMAT_CSV_H__